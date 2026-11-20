param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Start',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskInstall = Join-Path $taskRoot '.local-model'
$taskManifestPath = Join-Path $taskInstall 'manifest.json'
$taskStatePath = Join-Path $taskInstall 'server.json'
$taskEndpoint = "http://127.0.0.1:$Port"

function Get-LocalBlueHealth {
    try {
        $taskHealth = Invoke-RestMethod -Uri "$taskEndpoint/health" -TimeoutSec 2
        $taskModels = Invoke-RestMethod -Uri "$taskEndpoint/v1/models" -TimeoutSec 2
        if ($taskHealth.status -eq 'ok' -and $taskModels.data.id -contains 'LiquidAI/LFM3-1.7B') {
            return $true
        }
    } catch { }
    return $false
}

if ($Action -eq 'Status') {
    if (Get-LocalBlueHealth) { Write-Output "Local Blue ready: $taskEndpoint/v1" }
    else { Write-Output 'Local Blue is not ready.'; exit 1 }
    exit 0
}

if ($Action -eq 'Stop') {
    if (-not (Test-Path -LiteralPath $taskStatePath)) {
        Write-Output 'No managed Local Blue process recorded.'
        exit 0
    }
    $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
    $taskProcess = Get-Process -Id $taskState.process_id -ErrorAction SilentlyContinue
    # A recycled PID must never stop an unrelated application.
    if ($taskProcess -and $taskProcess.Path -eq $taskState.executable -and
        $taskProcess.StartTime.ToUniversalTime().ToString('o') -eq $taskState.started_at) {
        Stop-Process -Id $taskProcess.Id
        Write-Output 'Stopped Local Blue.'
    } else { Write-Output 'The recorded Local Blue process is no longer running.' }
    Remove-Item -LiteralPath $taskStatePath
    exit 0
}

if (Get-LocalBlueHealth) {
    Write-Output "Local Blue already ready: $taskEndpoint/v1"
    exit 0
}
if (-not (Test-Path -LiteralPath $taskManifestPath)) {
    throw 'Install first: .\.venv\Scripts\python.exe scripts/install_local_blue.py'
}
$taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw | ConvertFrom-Json
$taskExecutable = $taskManifest.runtime_path
$taskModel = $taskManifest.model_path
if (-not (Test-Path -LiteralPath $taskExecutable) -or -not (Test-Path -LiteralPath $taskModel)) {
    throw 'Local runtime/model is missing; run scripts/install_local_blue.py again.'
}
$taskLayers = if ($taskManifest.runtime_backend -eq 'cpu') { '0' } else { '99' }
$taskArguments = @(
    '--model', ('"' + $taskModel + '"'),
    '--alias', 'LiquidAI/LFM3-1.7B', '--host', '127.0.0.1', '--port', "$Port",
    '--ctx-size', '4096', '--parallel', '1', '--threads', '6',
    '--gpu-layers', $taskLayers, '--jinja', '--reasoning', 'off',
    '--reasoning-budget', '0', '--temp', '0.1', '--top-k', '50',
    '--repeat-penalty', '1.1', '--n-predict', '768', '--no-webui'
)
if ($taskManifest.runtime_backend -eq 'vulkan') {
    # On this machine Vulkan0 is the NVIDIA GPU; avoid splitting onto Intel iGPU.
    $taskArguments += @('--device', 'Vulkan0')
}
$taskStdout = Join-Path $taskInstall 'server.stdout.log'
$taskStderr = Join-Path $taskInstall 'server.stderr.log'
$taskProcess = Start-Process -FilePath $taskExecutable -ArgumentList $taskArguments `
    -WorkingDirectory (Split-Path -Parent $taskExecutable) -WindowStyle Hidden `
    -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
@{
    process_id = $taskProcess.Id
    executable = $taskExecutable
    started_at = $taskProcess.StartTime.ToUniversalTime().ToString('o')
    port = $Port
} | ConvertTo-Json | Set-Content -LiteralPath $taskStatePath -Encoding UTF8

Write-Output "Loading Local Blue (PID $($taskProcess.Id))..."
for ($taskAttempt = 0; $taskAttempt -lt 60; $taskAttempt++) {
    Start-Sleep -Seconds 2
    if (Get-LocalBlueHealth) {
        Write-Output "Local Blue ready: $taskEndpoint/v1"
        exit 0
    }
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) {
        Get-Content -LiteralPath $taskStderr -Tail 20
        throw 'Local model server exited; inspect .local-model/server.stderr.log.'
    }
}
throw 'Model startup timed out; inspect .local-model/server.stderr.log.'
