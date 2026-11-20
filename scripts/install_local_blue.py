"""Install official llama.cpp and LiquidAI GGUF assets, verifying SHA-256.

Run: .venv\Scripts\python.exe scripts/install_local_blue.py
Sources: https://github.com/ggml-org/llama.cpp/releases/tag/b12559
         https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF
No global Python packages, CUDA toolkit or administrator rights are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INSTALL_DIR = ROOT / ".local-model"
VERSION = "b12559"
MODEL_NAME = "LFM3-1.7B-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/{MODEL_NAME}"
MODEL_SHA256 = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
RUNTIME_SHA256 = {
    "vulkan": "8456c8e5e95800017e7ab77e97be6445b0962faa962aadbba9a0a8159560a02e",
    "cpu": "2f3dcb3aea4f52bf734f52040d2a688bdd76c2f762ae200a159646b37d84e393",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha256(path) == expected_sha256:
        print(f"Already verified: {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Day11-local-model-installer"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            # A fresh URL avoids reusing an expired Hugging Face download link.
            request_url = url + ("&" if "?" in url else "?") + f"download=1&attempt={time.time_ns()}"
            with urlopen(Request(request_url, headers=headers), timeout=60) as response:
                resume = offset > 0 and response.status == 206
                if not resume:
                    offset = 0
                size = int(response.headers.get("Content-Length", "0")) + offset
                done, last_print = offset, 0.0
                with partial.open("ab" if resume else "wb") as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last_print >= 10:
                            print(f"{path.name}: {done / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)
                            last_print = time.monotonic()
            if sha256(partial) != expected_sha256:
                raise RuntimeError(f"SHA-256 mismatch for {path.name}; refusing to install.")
            partial.replace(path)
            print(f"SHA-256 verified: {path.name}", flush=True)
            return
        except RuntimeError:
            raise
        except Exception as error:
            print(f"Download attempt {attempt + 1} failed: {type(error).__name__}", flush=True)
            if attempt == 2:
                raise


def install(backend: str) -> None:
    filename = f"llama-{VERSION}-bin-win-{backend}-x64.zip"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{VERSION}/{filename}"
    archive = INSTALL_DIR / "downloads" / filename
    download(url, archive, RUNTIME_SHA256[backend])
    runtime = INSTALL_DIR / "runtime" / f"{VERSION}-{backend}"
    runtime.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (runtime / item.filename).resolve().is_relative_to(runtime.resolve()):
                raise RuntimeError("Archive contains a path outside the runtime directory.")
        bundle.extractall(runtime)
    servers = list(runtime.rglob("llama-server.exe"))
    if not servers:
        raise RuntimeError("Official runtime archive did not contain llama-server.exe.")
    model = INSTALL_DIR / "models" / MODEL_NAME
    download(MODEL_URL, model, MODEL_SHA256)
    manifest = {
        "model_id": "LiquidAI/LFM3-1.7B",
        "gguf_repository": "LiquidAI/LFM3-1.7B-GGUF",
        "quantization": "Q4_K_M",
        "model_path": str(model),
        "model_sha256": MODEL_SHA256,
        "model_url": MODEL_URL,
        "runtime_version": VERSION,
        "runtime_backend": backend,
        "runtime_path": str(servers[0]),
        "runtime_sha256": RUNTIME_SHA256[backend],
        "runtime_url": url,
    }
    (INSTALL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Installed. Start with: .\\scripts\\local_blue.ps1", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(RUNTIME_SHA256), default="vulkan")
    install(parser.parse_args().backend)
