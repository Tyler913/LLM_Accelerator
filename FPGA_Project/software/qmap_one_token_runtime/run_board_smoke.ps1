param(
    [ValidateSet("model", "control", "generate")]
    [string]$Mode = "model",
    [string]$Xsdb = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Xsdb)) {
    if (-not [string]::IsNullOrWhiteSpace($env:XILINX_VITIS)) {
        $Xsdb = Join-Path $env:XILINX_VITIS "bin\xsdb.bat"
    } else {
        $Xsdb = "D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis\bin\xsdb.bat"
    }
}

if (-not (Test-Path -LiteralPath $Xsdb -PathType Leaf)) {
    throw "XSDB was not found at '$Xsdb'. Pass -Xsdb or set XILINX_VITIS."
}

$launcher = Join-Path $PSScriptRoot "launch_qwen3_board.tcl"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Board launcher is missing: $launcher"
}

$previousMode = $env:QOT_BOARD_MODE
try {
    $env:QOT_BOARD_MODE = $Mode

    # AMD's Windows xsdb.bat ends with `endlocal` and discards the exit code
    # returned by loader.bat. Invoke the adjacent loader directly so an
    # uncaught Tcl error cannot be reported to the acceptance harness as a
    # successful launcher exit.
    $xsdbCommand = $Xsdb
    $xsdbArguments = @($launcher)
    if ([System.IO.Path]::GetFileName($Xsdb) -ieq "xsdb.bat") {
        $loader = Join-Path (Split-Path -Parent $Xsdb) "loader.bat"
        if (-not (Test-Path -LiteralPath $loader -PathType Leaf)) {
            throw "AMD loader.bat was not found next to '$Xsdb'."
        }
        $xsdbCommand = $loader
        $xsdbArguments = @("-exec", "xsdb", $launcher)
    }

    & $xsdbCommand @xsdbArguments
    $xsdbExitCode = $LASTEXITCODE
    if ($xsdbExitCode -ne 0) {
        throw "XSDB board launch failed with exit code $xsdbExitCode."
    }
} finally {
    $env:QOT_BOARD_MODE = $previousMode
}
