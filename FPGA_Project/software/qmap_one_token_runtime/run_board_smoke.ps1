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
    & $Xsdb $launcher
    if ($LASTEXITCODE -ne 0) {
        throw "XSDB board launch failed with exit code $LASTEXITCODE."
    }
} finally {
    $env:QOT_BOARD_MODE = $previousMode
}
