[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [int]$TokenId = -1,
    [string]$CondaEnvironment = "llm_fpga",
    [switch]$NoStalls
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @(& $Executable @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LogPath)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
    return ,$output
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tempRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "Temp"))
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $tempRoot "embedding_layer0_frontend_regression"
} elseif (![System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $repoRoot $RunRoot
}

$runRootFull = [System.IO.Path]::GetFullPath($RunRoot)
$tempRootPrefix = $tempRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar
if (($runRootFull -ne $tempRoot) -and
    !$runRootFull.StartsWith($tempRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunRoot must stay under $tempRoot"
}

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $runRootFull $runId
$vectorDir = Join-Path $sessionDir "vectors"
$xsimDir = Join-Path $sessionDir "xsim"
$pycacheDir = Join-Path $sessionDir "pycache"
New-Item -ItemType Directory -Force -Path $vectorDir,$xsimDir,$pycacheDir | Out-Null

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$tbPath = Join-Path $simDir "tb_qmap_one_token_embedding_layer0_frontend.sv"
$exporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\50_export_embedding_layer0_frontend_chain.py"
$timingChecker = Join-Path $simDir "check_embedding_layer0_frontend_timing.py"
$sourceVectorDir = Join-Path $simDir "vectors"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs

$conda = (Get-Command conda -ErrorAction Stop).Source
$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("embedding -> Layer 0 RMSNorm -> QKV regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")

$savedPycachePrefix = $env:PYTHONPYCACHEPREFIX
$env:PYTHONPYCACHEPREFIX = $pycacheDir
Push-Location $sessionDir
try {
    $exportArgs = @("run", "--no-capture-output", "-n", $CondaEnvironment,
        "python", "-B", $exporter, "--output-dir", $vectorDir)
    if ($TokenId -ge 0) {
        $exportArgs += @("--token-id", "$TokenId")
    }
    Write-Host "[export]  tied-Q4 embedding -> Layer 0 RMSNorm -> full QKV"
    $exportOutput = Invoke-LoggedCommand -Executable $conda -Arguments $exportArgs `
        -LogPath (Join-Path $sessionDir "export.log")
    if (($exportOutput -join "`n") -notmatch
        "PASS: exported tied-Q4 embedding -> Layer 0 input RMSNorm -> full QKV chain") {
        throw "Frontend chain exporter did not print its PASS marker"
    }
    $summary.Add("PASS vector_export")

    New-Item -ItemType Junction -Path (Join-Path $xsimDir "vectors") `
        -Target $vectorDir | Out-Null
    New-Item -ItemType Junction -Path (Join-Path $xsimDir "source_vectors") `
        -Target $sourceVectorDir | Out-Null

    Push-Location $xsimDir
    try {
        $snapshot = "embedding_layer0_frontend_xsim"
        $xvlogArgs = @("--sv", "--relax")
        foreach ($includeDir in $rtlIncludeDirs) {
            $xvlogArgs += @("-i", $includeDir)
        }
        $xvlogArgs += @("--log", (Join-Path $xsimDir "xvlog.log")) + $rtlFiles + @($tbPath)
        Write-Host "[xvlog]  compile focused frontend chain"
        Invoke-LoggedCommand -Executable $xvlog -Arguments $xvlogArgs `
            -LogPath (Join-Path $xsimDir "xvlog_console.log") | Out-Null

        $xelabArgs = @("--relax", "--snapshot", $snapshot,
            "--log", (Join-Path $xsimDir "xelab.log"),
            "tb_qmap_one_token_embedding_layer0_frontend")
        Write-Host "[xelab]  elaborate focused frontend chain"
        Invoke-LoggedCommand -Executable $xelab -Arguments $xelabArgs `
            -LogPath (Join-Path $xsimDir "xelab_console.log") | Out-Null

        $xsimArgs = @($snapshot)
        if ($NoStalls) {
            $xsimArgs += @("-testplusarg", "nostall")
        }
        $xsimArgs += @("--log", (Join-Path $xsimDir "xsim.log"), "--runall")
        Write-Host "[xsim]   run focused frontend chain"
        $xsimOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs `
            -LogPath (Join-Path $xsimDir "xsim_console.log")
        $combinedOutput = $xsimOutput -join "`n"
        if ($combinedOutput -match "(?m)^FAIL:") {
            throw "XSim printed a FAIL line. See $(Join-Path $xsimDir 'xsim.log')"
        }
        if ($combinedOutput -notmatch
            "(?m)^PASS: tied-Q4 embedding fed Layer 0 input RMSNorm and full QKV exactly\.$") {
            throw "XSim did not print the focused frontend PASS marker"
        }
        $passLine = $xsimOutput | Where-Object { $_ -match "^PASS:" } | Select-Object -Last 1
        $summary.Add("PASS xsim :: $passLine")
    } finally {
        Pop-Location
    }

    $auditPath = Join-Path $xsimDir "timing_audit.json"
    $auditArgs = @(
        "run", "--no-capture-output", "-n", $CondaEnvironment,
        "python", "-B", $timingChecker,
        (Join-Path $xsimDir "timing_trace.csv"),
        "--chain-manifest", (Join-Path $vectorDir "chain_manifest.json"),
        "--qkv-manifest", (Join-Path $vectorDir "layer0_qkv_from_embedding_rmsnorm_manifest.json"),
        "--xsim-log", (Join-Path $xsimDir "xsim.log"),
        "--output", $auditPath
    )
    Write-Host "[audit]  verify exact counts and producer/consumer ordering"
    $auditOutput = Invoke-LoggedCommand -Executable $conda -Arguments $auditArgs `
        -LogPath (Join-Path $xsimDir "timing_audit_console.log")
    if (($auditOutput -join "`n") -notmatch
        "PASS: embedding -> Layer 0 RMSNorm -> QKV timing and counts are exact\.") {
        throw "Timing auditor did not print its PASS marker"
    }
    $summary.Add("PASS timing_audit :: $auditPath")

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All vectors, compiler products, logs, traces, and audits are under $sessionDir"
} finally {
    Pop-Location
    $env:PYTHONPYCACHEPREFIX = $savedPycachePrefix
}
