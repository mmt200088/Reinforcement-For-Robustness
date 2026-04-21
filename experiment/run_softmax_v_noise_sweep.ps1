param(
    [switch]$Foreground,
    [string]$OutputDir = "experiment/outputs/noise/softmax_v_sweep",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

function Quote-PSArg {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

if (-not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = "0"
}

$hasOutputDir = $false
foreach ($arg in $PythonArgs) {
    if ($arg -eq "--output_dir") {
        $hasOutputDir = $true
        break
    }
}
if (-not $hasOutputDir) {
    $PythonArgs = @($PythonArgs) + @("--output_dir", $OutputDir)
}

if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDirPath = $OutputDir
} else {
    $OutputDirPath = Join-Path $RepoRoot $OutputDir
}

New-Item -ItemType Directory -Force -Path $OutputDirPath | Out-Null
$LogPath = Join-Path $OutputDirPath "run.log"
$PidPath = Join-Path $OutputDirPath "pid.txt"

$ModuleArgs = @("-u", "-m", "experiment.scripts.noise.softmax_v_noise_sweep") + $PythonArgs

if ($Foreground) {
    Set-Location -LiteralPath $RepoRoot
    & python @ModuleArgs
    exit $LASTEXITCODE
}

$quotedRepo = Quote-PSArg $RepoRoot
$quotedLog = Quote-PSArg $LogPath
$quotedArgs = ($ModuleArgs | ForEach-Object { Quote-PSArg $_ }) -join " "
$cudaValue = Quote-PSArg $env:CUDA_VISIBLE_DEVICES

$command = @"
Set-Location -LiteralPath $quotedRepo
`$env:CUDA_VISIBLE_DEVICES = $cudaValue
python $quotedArgs *> $quotedLog
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$proc = Start-Process `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $PidPath -Value $proc.Id -Encoding UTF8

Write-Host "Experiments started in background."
Write-Host "  PID:  $($proc.Id)"
Write-Host "  Log:  $LogPath"
Write-Host "  Check: Get-Content -Wait $LogPath"
Write-Host "  Stop:  Stop-Process -Id (Get-Content $PidPath)"
