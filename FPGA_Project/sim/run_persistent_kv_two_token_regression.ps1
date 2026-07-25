[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [switch]$SkipXsim
)

$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @(& $Executable @Arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LogPath)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
    return ,$output
}

function Assert-PersistentKvPass {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [AllowEmptyCollection()]
        [string[]]$Output,
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [Parameter(Mandatory = $true)]
        [string]$SimulatorName
    )

    $joined = $Output -join "`n"
    if ($joined -notmatch "(?m)^PASS: persistent KV frontend\+score two-token proof passed;") {
        throw "$SimulatorName did not print the persistent-KV PASS marker. See $LogPath"
    }
    if ($joined -match "(?m)^FAIL:") {
        throw "$SimulatorName printed a FAIL line. See $LogPath"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tempRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "Temp"))
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $tempRoot "persistent_kv_two_token_regression"
}
elseif (![System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $repoRoot $RunRoot
}

$runRootFull = [System.IO.Path]::GetFullPath($RunRoot)
$tempPrefix = $tempRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
    [System.IO.Path]::DirectorySeparatorChar
if (($runRootFull -ne $tempRoot) -and
    !$runRootFull.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunRoot must stay under $tempRoot"
}

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $runRootFull $runId
New-Item -ItemType Directory -Force -Path $sessionDir | Out-Null

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$tbPath = Join-Path $simDir "tb_qmap_attention_persistent_kv_two_token.sv"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs
$top = "tb_qmap_attention_persistent_kv_two_token"
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("persistent KV two-token regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")
$summary.Add("rtl_sources=$($rtlFiles.Count)")

Push-Location $sessionDir
try {
    $iverilog = (Get-Command iverilog -ErrorAction Stop).Source
    $vvp = (Get-Command vvp -ErrorAction Stop).Source
    $vvpImage = Join-Path $sessionDir "$top.vvp"
    $iverilogCompileLog = Join-Path $sessionDir "iverilog_compile.log"
    $iverilogRunLog = Join-Path $sessionDir "iverilog_run.log"
    $iverilogArgs = @("-g2012", "-Wall")
    foreach ($includeDir in $rtlIncludeDirs) {
        $iverilogArgs += @("-I", $includeDir)
    }
    $iverilogArgs += @("-s", $top, "-o", $vvpImage)
    $iverilogArgs += $rtlFiles
    $iverilogArgs += @($tbPath)

    Write-Host "[iverilog] compile $top from the RTL manifest"
    Invoke-LoggedCommand -Executable $iverilog -Arguments $iverilogArgs `
        -LogPath $iverilogCompileLog | Out-Null
    Write-Host "[vvp]      run persistent-KV proof"
    $iverilogOutput = Invoke-LoggedCommand -Executable $vvp -Arguments @($vvpImage) `
        -LogPath $iverilogRunLog
    Assert-PersistentKvPass -Output $iverilogOutput -LogPath $iverilogRunLog `
        -SimulatorName "Icarus"
    $summary.Add("PASS iverilog :: $iverilogRunLog")

    if ($SkipXsim) {
        $summary.Add("SKIP xsim :: requested with -SkipXsim")
    }
    else {
        $xvlogCommand = Get-Command xvlog -ErrorAction SilentlyContinue
        $xelabCommand = Get-Command xelab -ErrorAction SilentlyContinue
        $xsimCommand = Get-Command xsim -ErrorAction SilentlyContinue
        if (($null -eq $xvlogCommand) -or ($null -eq $xelabCommand) -or
            ($null -eq $xsimCommand)) {
            $summary.Add("SKIP xsim :: xvlog/xelab/xsim not available on PATH")
        }
        else {
            $snapshot = "persistent_kv_two_token_xsim"
            $xvlogLog = Join-Path $sessionDir "xvlog.log"
            $xelabLog = Join-Path $sessionDir "xelab.log"
            $xsimLog = Join-Path $sessionDir "xsim.log"
            $xvlogArgs = @("--sv", "--relax")
            foreach ($includeDir in $rtlIncludeDirs) {
                $xvlogArgs += @("-i", $includeDir)
            }
            $xvlogArgs += @("--log", $xvlogLog)
            $xvlogArgs += $rtlFiles
            $xvlogArgs += @($tbPath)

            Write-Host "[xvlog]    compile $top from the RTL manifest"
            Invoke-LoggedCommand -Executable $xvlogCommand.Source -Arguments $xvlogArgs `
                -LogPath (Join-Path $sessionDir "xvlog_console.log") | Out-Null
            $xelabArgs = @(
                "--relax",
                "--debug", "typical",
                "--snapshot", $snapshot,
                "--log", $xelabLog,
                $top
            )
            Write-Host "[xelab]    elaborate $snapshot"
            Invoke-LoggedCommand -Executable $xelabCommand.Source -Arguments $xelabArgs `
                -LogPath (Join-Path $sessionDir "xelab_console.log") | Out-Null
            $xsimArgs = @($snapshot, "--log", $xsimLog, "--runall")
            Write-Host "[xsim]     run persistent-KV proof"
            $xsimOutput = Invoke-LoggedCommand -Executable $xsimCommand.Source `
                -Arguments $xsimArgs -LogPath (Join-Path $sessionDir "xsim_console.log")
            Assert-PersistentKvPass -Output $xsimOutput -LogPath $xsimLog `
                -SimulatorName "XSim"
            $summary.Add("PASS xsim :: $xsimLog")
        }
    }

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All logs and simulator intermediates are under $sessionDir"
}
finally {
    Pop-Location
}
