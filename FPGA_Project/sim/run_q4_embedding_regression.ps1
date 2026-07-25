[CmdletBinding()]
param(
    [string]$RunRoot = "",
    [int]$TokenId = -1,
    [string]$CondaEnvironment = "llm_fpga",
    [switch]$SkipXsim
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
    $RunRoot = Join-Path $tempRoot "q4_embedding_regression"
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
New-Item -ItemType Directory -Force -Path $vectorDir | Out-Null

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$exporter = Join-Path $repoRoot "Qwen3-0.6B-Base\python_each_module\49_export_q4_embedding_vectors.py"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs
$tests = @(
    @{ Top = "tb_q4_embedding_lookup"; Source = "tb_q4_embedding_lookup.sv" },
    @{ Top = "tb_qmap_one_token_axil_embedding_top"; Source = "tb_qmap_one_token_axil_embedding_top.sv" },
    @{ Top = "tb_qmap_one_token_axi_embedding_top"; Source = "tb_qmap_one_token_axi_embedding_top.sv" },
    @{ Top = "tb_qmap_one_token_axi_embedding_top_mem_error"; Source = "tb_qmap_one_token_axi_embedding_top.sv" }
)

$conda = (Get-Command conda -ErrorAction Stop).Source
$iverilog = (Get-Command iverilog -ErrorAction Stop).Source
$vvp = (Get-Command vvp -ErrorAction Stop).Source
if (!$SkipXsim) {
    $xvlog = (Get-Command xvlog -ErrorAction Stop).Source
    $xelab = (Get-Command xelab -ErrorAction Stop).Source
    $xsim = (Get-Command xsim -ErrorAction Stop).Source
}
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("Q4 embedding local regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")

Push-Location $sessionDir
try {
    $exportLog = Join-Path $sessionDir "export.log"
    $exportArgs = @("run", "--no-capture-output", "-n", $CondaEnvironment,
        "python", $exporter, "--output-dir", $vectorDir)
    if ($TokenId -ge 0) {
        $exportArgs += @("--token-id", "$TokenId")
    }
    Write-Host "[export]  tied Q4 embedding vectors"
    $exportOutput = Invoke-LoggedCommand -Executable $conda -Arguments $exportArgs -LogPath $exportLog
    if (($exportOutput -join "`n") -notmatch "Exported tied Q4 embedding lookup vectors") {
        throw "Embedding exporter did not print its success marker. See $exportLog"
    }
    $summary.Add("PASS embedding_vector_export")

    $vectorArg = "+VECTOR_DIR=" + $vectorDir.Replace('\', '/')
    foreach ($test in $tests) {
        $name = $test.Top
        $testDir = Join-Path $sessionDir $name
        New-Item -ItemType Directory -Force -Path $testDir | Out-Null

        $vvpPath = Join-Path $testDir "$name.vvp"
        $compileLog = Join-Path $testDir "compile.log"
        $runLog = Join-Path $testDir "run.log"
        $tbPath = Join-Path $simDir $test.Source
        $compileArgs = @("-g2012", "-Wall")
        foreach ($includeDir in $rtlIncludeDirs) {
            $compileArgs += @("-I", $includeDir)
        }
        $compileArgs += @("-s", $name, "-o", $vvpPath) + $rtlFiles + @($tbPath)

        Write-Host "[compile] $name"
        Invoke-LoggedCommand -Executable $iverilog -Arguments $compileArgs -LogPath $compileLog | Out-Null

        Write-Host "[run]     $name"
        $runOutput = Invoke-LoggedCommand -Executable $vvp -Arguments @($vvpPath, $vectorArg) -LogPath $runLog
        if (($runOutput -join "`n") -notmatch "(?m)^PASS[: ]") {
            throw "Simulation did not print a PASS line. See $runLog"
        }

        $passLine = $runOutput | Where-Object { $_ -match "^PASS[: ]" } | Select-Object -Last 1
        $summary.Add("PASS $name :: $passLine")
    }

    if (!$SkipXsim) {
        $name = "tb_qmap_one_token_axi_embedding_top_xsim"
        $xsimDir = Join-Path $sessionDir $name
        New-Item -ItemType Directory -Force -Path $xsimDir | Out-Null
        New-Item -ItemType Junction -Path (Join-Path $xsimDir "vectors") `
            -Target $vectorDir | Out-Null
        $snapshot = "q4_embedding_axi_top_xsim"
        $tbPath = Join-Path $simDir "tb_qmap_one_token_axi_embedding_top.sv"

        Push-Location $xsimDir
        try {
            $xvlogArgs = @("--sv", "--relax")
            foreach ($includeDir in $rtlIncludeDirs) {
                $xvlogArgs += @("-i", $includeDir)
            }
            $xvlogArgs += @("--log", (Join-Path $xsimDir "xvlog.log")) + $rtlFiles + @($tbPath)
            Write-Host "[xvlog]  $name"
            Invoke-LoggedCommand -Executable $xvlog -Arguments $xvlogArgs `
                -LogPath (Join-Path $xsimDir "xvlog_console.log") | Out-Null

            $xelabArgs = @("--relax", "--snapshot", $snapshot,
                "--log", (Join-Path $xsimDir "xelab.log"),
                "tb_qmap_one_token_axi_embedding_top")
            Write-Host "[xelab]  $name"
            Invoke-LoggedCommand -Executable $xelab -Arguments $xelabArgs `
                -LogPath (Join-Path $xsimDir "xelab_console.log") | Out-Null

            $xsimArgs = @($snapshot, "--log", (Join-Path $xsimDir "xsim.log"),
                "--runall")
            Write-Host "[xsim]   $name"
            $xsimOutput = Invoke-LoggedCommand -Executable $xsim -Arguments $xsimArgs `
                -LogPath (Join-Path $xsimDir "xsim_console.log")
            $combinedOutput = $xsimOutput -join "`n"
            if ($combinedOutput -notmatch "(?m)^PASS:") {
                throw "XSim did not print a PASS line. See $(Join-Path $xsimDir 'xsim.log')"
            }
            if ($combinedOutput -match "(?m)^FAIL:") {
                throw "XSim printed a FAIL line. See $(Join-Path $xsimDir 'xsim.log')"
            }
            $passLine = $xsimOutput | Where-Object { $_ -match "^PASS:" } | Select-Object -Last 1
            $summary.Add("PASS $name :: $passLine")
        } finally {
            Pop-Location
        }
    }

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All vectors, intermediates, and logs are under $sessionDir"
} finally {
    Pop-Location
}
