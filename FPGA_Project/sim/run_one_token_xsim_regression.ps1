[CmdletBinding()]
param(
    [ValidateSet("l1_l2", "true3", "all")]
    [string]$Scenario = "true3",
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$ConsoleLogPath
    )

    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @(& $Executable @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $ConsoleLogPath)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference

    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $ConsoleLogPath"
    }

    return ,$output
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tempRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "Temp"))
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $tempRoot "one_token_xsim_regression"
} elseif (![System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $repoRoot $RunRoot
}

$runRootFull = [System.IO.Path]::GetFullPath($RunRoot)
$tempRootPrefix = $tempRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (($runRootFull -ne $tempRoot) -and
    !$runRootFull.StartsWith($tempRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunRoot must stay under $tempRoot"
}

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $runRootFull $runId
New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null

# The testbench reads checked-in vectors through repo-relative paths. These
# junctions expose those inputs while every generated file remains in Temp.
New-Item -ItemType Junction -Path (Join-Path $sessionDir "FPGA_Project") -Target (Join-Path $repoRoot "FPGA_Project") | Out-Null
New-Item -ItemType Junction -Path (Join-Path $sessionDir "artifacts") -Target (Join-Path $repoRoot "artifacts") | Out-Null

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$tbPath = Join-Path $repoRoot "FPGA_Project\sim\tb_qmap_one_token_layer_scheduler.sv"
$timingChecker = Join-Path $simDir "check_one_token_timing_trace.py"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs

$xvlog = (Get-Command xvlog -ErrorAction Stop).Source
$xelab = (Get-Command xelab -ErrorAction Stop).Source
$xsim = (Get-Command xsim -ErrorAction Stop).Source
$conda = (Get-Command conda -ErrorAction Stop).Source
$snapshot = "qmap_one_token_axil_top_to_tail_xsim"
$defines = @(
    "QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL",
    "QMAP_ONE_TOKEN_TB_USE_TOP",
    "QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL",
    "QMAP_ONE_TOKEN_TB_USE_AXIL_TOP"
)

$selectedScenarios = switch ($Scenario) {
    "l1_l2" { @(@{ Name = "l1_l2"; PlusArg = "l1_l2_mmio_top_tail_only" }) }
    "true3" { @(@{ Name = "true3"; PlusArg = "true3_mmio_top_tail_only" }) }
    "all" {
        @(
            @{ Name = "l1_l2"; PlusArg = "l1_l2_mmio_top_tail_only" },
            @{ Name = "true3"; PlusArg = "true3_mmio_top_tail_only" }
        )
    }
}

$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("one-token xsim regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")

Push-Location $sessionDir
try {
    $xvlogArgs = @("--sv", "--relax")
    foreach ($includeDir in $rtlIncludeDirs) {
        $xvlogArgs += @("-i", $includeDir)
    }
    foreach ($define in $defines) {
        $xvlogArgs += @("-d", $define)
    }
    $xvlogArgs += @("--log", (Join-Path $sessionDir "xvlog.log"))
    $xvlogArgs += $rtlFiles
    $xvlogArgs += @($tbPath)

    Write-Host "[xvlog]  compile one-token AXI-Lite top-to-tail snapshot"
    Invoke-LoggedCommand -Executable $xvlog -Arguments $xvlogArgs -ConsoleLogPath (Join-Path $sessionDir "xvlog_console.log") | Out-Null

    $xelabArgs = @(
        "--relax",
        "--debug", "typical",
        "--snapshot", $snapshot,
        "--log", (Join-Path $sessionDir "xelab.log"),
        "tb_qmap_one_token_layer_scheduler"
    )
    Write-Host "[xelab]  elaborate $snapshot"
    Invoke-LoggedCommand -Executable $xelab -Arguments $xelabArgs -ConsoleLogPath (Join-Path $sessionDir "xelab_console.log") | Out-Null

    foreach ($item in $selectedScenarios) {
        $scenarioDir = Join-Path $sessionDir $item.Name
        New-Item -ItemType Directory -Force -Path $scenarioDir | Out-Null
        $tracePath = Join-Path $scenarioDir "timing_trace.csv"
        $xsimLog = Join-Path $scenarioDir "xsim.log"
        $xsimArgs = @(
            $snapshot,
            "-testplusarg", $item.PlusArg,
            "-testplusarg", "fastmem",
            "-testplusarg", "progress",
            "-testplusarg", "trace_to_cwd",
            "--log", $xsimLog,
            "--runall"
        )

        Write-Host "[xsim]   $($item.Name) with timing trace"
        $runOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs -ConsoleLogPath (Join-Path $scenarioDir "xsim_console.log")
        $combinedOutput = $runOutput -join "`n"
        if ($combinedOutput -notmatch "(?m)^PASS:") {
            throw "XSim did not print a PASS line. See $xsimLog"
        }
        if ($combinedOutput -match "(?m)^FAIL:") {
            throw "XSim printed a FAIL line. See $xsimLog"
        }
        if (!(Test-Path -LiteralPath $tracePath) -or ((Get-Item -LiteralPath $tracePath).Length -eq 0)) {
            throw "XSim did not create a non-empty timing trace at $tracePath"
        }

        $auditPath = Join-Path $scenarioDir "timing_audit.json"
        $auditArgs = @(
            "run", "-n", "llm_fpga", "python", $timingChecker,
            $tracePath,
            "--scenario", $item.Name,
            "--xsim-log", $xsimLog,
            "--output", $auditPath
        )
        Write-Host "[audit]  $($item.Name) timing trace"
        $auditOutput = Invoke-LoggedCommand -Executable $conda -Arguments $auditArgs -ConsoleLogPath (Join-Path $scenarioDir "timing_audit_console.log")
        if (($auditOutput -join "`n") -notmatch "PASS: one-token timing trace ordering and counts are exact\.") {
            throw "Timing audit did not print its PASS line. See $auditPath"
        }

        $passLine = $runOutput | Where-Object { $_ -match "^PASS:" } | Select-Object -Last 1
        $summary.Add("PASS $($item.Name) :: $passLine")
        $summary.Add("TRACE $($item.Name) :: $tracePath")
        $summary.Add("AUDIT $($item.Name) :: $auditPath")
    }

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All XSim intermediates, logs, and traces are under $sessionDir"
} finally {
    Pop-Location
}
