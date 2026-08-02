[CmdletBinding()]
param(
    [string]$RunRoot = ""
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
    $RunRoot = Join-Path $tempRoot "one_token_local_regression"
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

$rtlDir = Join-Path $repoRoot "FPGA_Project\rtl"
$simDir = Join-Path $repoRoot "FPGA_Project\sim"
$runtimeDir = Join-Path $repoRoot "FPGA_Project\software\qmap_one_token_runtime"
. (Join-Path $simDir "load_rtl_manifest.ps1")
$rtlBuild = Get-RtlBuildManifest -RtlDir $rtlDir
$rtlFiles = $rtlBuild.SourceFiles
$rtlIncludeDirs = $rtlBuild.IncludeDirs

$tests = @(
    @{ Top = "tb_axi4_read_master"; Source = "tb_axi4_read_master.sv" },
    @{ Top = "tb_axi4_write_master"; Source = "tb_axi4_write_master.sv" },
    @{ Top = "tb_axi4lite_to_mmio_regs"; Source = "tb_axi4lite_to_mmio_regs.sv" },
    @{ Top = "tb_qmap_one_token_control_regs"; Source = "tb_qmap_one_token_control_regs.sv" },
    @{ Top = "tb_qmap_one_token_control_regs_bram"; Source = "tb_qmap_one_token_control_regs_bram.sv" },
    @{ Top = "tb_qmap_one_token_layer_scheduler_validation"; Source = "tb_qmap_one_token_layer_scheduler_validation.sv" },
    @{ Top = "tb_qmap_one_token_mmio_top"; Source = "tb_qmap_one_token_mmio_top.sv" },
    @{ Top = "tb_qmap_one_token_axil_top"; Source = "tb_qmap_one_token_axil_top.sv" },
    @{ Top = "tb_qmap_one_token_axi_top"; Source = "tb_qmap_one_token_axi_top.sv" }
)

$iverilog = (Get-Command iverilog -ErrorAction Stop).Source
$vvp = (Get-Command vvp -ErrorAction Stop).Source
$gcc = (Get-Command gcc -ErrorAction Stop).Source
$summary = [System.Collections.Generic.List[string]]::new()
$summary.Add("one-token local regression")
$summary.Add("run_id=$runId")
$summary.Add("session_dir=$sessionDir")

Push-Location $sessionDir
try {
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
        $runOutput = Invoke-LoggedCommand -Executable $vvp -Arguments @($vvpPath) -LogPath $runLog
        if (($runOutput -join "`n") -notmatch "(?m)^PASS[: ]") {
            throw "Simulation did not print a PASS line. See $runLog"
        }

        $passLine = $runOutput | Where-Object { $_ -match "^PASS[: ]" } | Select-Object -Last 1
        $summary.Add("PASS $name :: $passLine")
    }

    $contractDir = Join-Path $sessionDir "qmap_one_token_register_contract"
    New-Item -ItemType Directory -Force -Path $contractDir | Out-Null
    $contractLog = Join-Path $contractDir "register_map.log"
    $svText = Get-Content -Raw (Join-Path $rtlDir "top\one_token\qmap_one_token_control_regs.sv")
    $headerText = Get-Content -Raw (Join-Path $runtimeDir "qmap_one_token_regs.h")
    $svMatches = [regex]::Matches($svText,
        "localparam\s+logic\s+\[9\s*:\s*0\]\s+REG_([A-Z0-9_]+)\s*=\s*10'h([0-9A-Fa-f]+)")
    $headerMatches = [regex]::Matches($headerText,
        "#define\s+QOT_REG_([A-Z0-9_]+)\s+0x([0-9A-Fa-f]+)u")
    $svOffsets = @{}
    $headerOffsets = @{}
    foreach ($match in $svMatches) {
        $svOffsets[$match.Groups[1].Value] =
            4 * [Convert]::ToUInt32($match.Groups[2].Value, 16)
    }
    foreach ($match in $headerMatches) {
        $headerOffsets[$match.Groups[1].Value] =
            [Convert]::ToUInt32($match.Groups[2].Value, 16)
    }
    $contractLines = [System.Collections.Generic.List[string]]::new()
    foreach ($name in ($svOffsets.Keys | Sort-Object)) {
        if (!$headerOffsets.ContainsKey($name)) {
            throw "Register $name exists in RTL but not qmap_one_token_regs.h"
        }
        if ($headerOffsets[$name] -ne $svOffsets[$name]) {
            throw ("Register {0} mismatch: RTL byte offset 0x{1:X3}, header 0x{2:X3}" -f
                $name, $svOffsets[$name], $headerOffsets[$name])
        }
        $contractLines.Add(("PASS {0}=0x{1:X3}" -f $name, $svOffsets[$name]))
    }
    foreach ($name in $headerOffsets.Keys) {
        if (!$svOffsets.ContainsKey($name)) {
            throw "Register $name exists in qmap_one_token_regs.h but not RTL"
        }
    }
    [System.IO.File]::WriteAllLines($contractLog, [string[]]$contractLines)
    Write-Host "[contract] qmap_one_token RTL/C register offsets"
    $summary.Add("PASS qmap_one_token_register_contract :: $($svOffsets.Count) offsets")

    $hostDir = Join-Path $sessionDir "qmap_one_token_runtime_host_syntax"
    New-Item -ItemType Directory -Force -Path $hostDir | Out-Null
    $hostLog = Join-Path $hostDir "gcc.log"
    $hostArgs = @(
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fsyntax-only",
        "-I$runtimeDir",
        (Join-Path $runtimeDir "main.c")
    )

    Write-Host "[syntax]  qmap_one_token_runtime/main.c"
    Invoke-LoggedCommand -Executable $gcc -Arguments $hostArgs -LogPath $hostLog | Out-Null
    $summary.Add("PASS qmap_one_token_runtime_host_syntax")

    $hostTestSource = Join-Path $runtimeDir "test_runtime_host.c"
    $hostTestExe = Join-Path $hostDir "test_runtime_host.exe"
    $hostTestCompileLog = Join-Path $hostDir "test_runtime_host_compile.log"
    $hostTestRunLog = Join-Path $hostDir "test_runtime_host_run.log"
    $hostTestArgs = @(
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I$runtimeDir",
        $hostTestSource,
        "-o",
        $hostTestExe
    )
    Write-Host "[compile] qmap_one_token_runtime/test_runtime_host.c"
    Invoke-LoggedCommand -Executable $gcc -Arguments $hostTestArgs -LogPath $hostTestCompileLog | Out-Null
    Write-Host "[run]     qmap_one_token_runtime/test_runtime_host.c"
    $hostTestOutput = Invoke-LoggedCommand -Executable $hostTestExe -Arguments @() -LogPath $hostTestRunLog
    if (($hostTestOutput -join "`n") -notmatch "(?m)^PASS[: ]") {
        throw "Host runtime test did not print a PASS line. See $hostTestRunLog"
    }
    $summary.Add("PASS qmap_one_token_runtime_host_decode")

    $summaryPath = Join-Path $sessionDir "summary.txt"
    [System.IO.File]::WriteAllLines($summaryPath, [string[]]$summary)
    $summary | ForEach-Object { Write-Host $_ }
    Write-Host "All intermediates and logs are under $sessionDir"
} finally {
    Pop-Location
}
