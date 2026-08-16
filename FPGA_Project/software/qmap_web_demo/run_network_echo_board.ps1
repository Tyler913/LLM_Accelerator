param(
    [string]$Workspace = "F:\vwn",
    [string]$Xsdb = ""
)

$ErrorActionPreference = "Stop"

$workspacePath = (Resolve-Path -LiteralPath $Workspace).Path
$workspacePrefix = $workspacePath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar

function Resolve-WorkspaceFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not [System.IO.File]::Exists($resolved)) {
        throw "$Label is not a regular file: $resolved"
    }
    if (-not $resolved.StartsWith(
        $workspacePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escapes the audited workspace: $resolved"
    }
    return $resolved
}

function Resolve-WorkspaceDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not [System.IO.Directory]::Exists($resolved)) {
        throw "$Label is not a directory: $resolved"
    }
    if (-not $resolved.StartsWith(
        $workspacePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escapes the audited workspace: $resolved"
    }
    return $resolved
}

$manifestPath = Join-Path $workspacePath "network_workspace_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Successful network workspace manifest is missing: $manifestPath"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$platformName = [string]$manifest.platform_name
$echoName = [string]$manifest.app_name
if ($platformName -notmatch '^[A-Za-z][A-Za-z0-9_]{0,31}$' -or
    $echoName -notmatch '^[A-Za-z][A-Za-z0-9_]{0,31}$') {
    throw "Network manifest contains an invalid component name"
}
if ($manifest.build_results.platform -ne 0 -or
    $manifest.build_results.application -ne 0) {
    throw "Network workspace manifest does not record successful builds"
}
if (@($manifest.bsp_libraries).Count -ne 1 -or
    $manifest.bsp_libraries[0] -ne "lwip220") {
    throw "Network workspace manifest lacks the audited lwIP BSP contract"
}
$expectedBspConfig = @(
    "lwip220|lwip220_api_mode|RAW_API",
    "lwip220|lwip220_dhcp|true",
    "lwip220|lwip220_ipv6_enable|false",
    "lwip220|lwip220_lwip_dhcp_does_acd_check|true",
    "lwip220|lwip220_pbuf_pool_size|2048",
    "xiltimer|XILTIMER_en_interval_timer|true"
) | Sort-Object
$actualBspConfig = @(
    $manifest.bsp_config | ForEach-Object {
        "{0}|{1}|{2}" -f $_.library, $_.parameter, $_.value
    }
) | Sort-Object
if (@(Compare-Object $expectedBspConfig $actualBspConfig).Count -ne 0) {
    throw "Network workspace manifest has a mismatched lwIP BSP configuration"
}

$overrides = @($manifest.bsp_library_overrides)
if ($overrides.Count -ne 1) {
    throw "Expected exactly one audited BSP library override"
}
$override = $overrides[0]
$expectedOriginalHash = "b45bad2d4c9e2543db7ec8e70b7b450633d748de749b8e67dc4f27894a63430d"
$expectedMetadataHash = "898d479f1ff6b828ab4666f371ca067e69b21b8b767d4b412422c18e8b09800e"
if ($override.library -ne "lwip220" -or
    $override.version -ne "lwip220_v1_2" -or
    $override.motorcomm_phy_id -ne "0x0000011A" -or
    ([string]$override.original_sha256).ToLowerInvariant() -ne $expectedOriginalHash -or
    ([string]$override.metadata_sha256).ToLowerInvariant() -ne $expectedMetadataHash) {
    throw "BSP override does not match the pinned YT8521 lwip220 contract"
}
$overrideDirectory = Resolve-WorkspaceDirectory `
    -Path ([string]$override.destination) `
    -Label "Staged lwip220 override"
$patchedRelative = ([string]$override.patched_file).Replace('/', '\')
if ($patchedRelative -ne "src\lwip-2.2.0\contrib\ports\xilinx\netif\xemacpsif_physpeed.c") {
    throw "BSP override has an unexpected patched source path"
}
$patchedSource = Resolve-WorkspaceFile `
    -Path (Join-Path $overrideDirectory $patchedRelative) `
    -Label "Staged YT8521 source"
$patchedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $patchedSource).Hash.ToLowerInvariant()
if ($patchedHash -ne ([string]$override.patched_sha256).ToLowerInvariant()) {
    throw "Staged YT8521 source SHA-256 mismatch"
}

$builtSourceExpected = Join-Path $workspacePath (
    "$platformName\psu_cortexa53_0\standalone_psu_cortexa53_0\bsp\libsrc\lwip220\$patchedRelative"
)
$builtSource = Resolve-WorkspaceFile `
    -Path ([string]$override.build_outputs.bsp_copied_source.path) `
    -Label "Vitis BSP copied YT8521 source"
if (-not [string]::Equals(
    $builtSource,
    (Resolve-Path -LiteralPath $builtSourceExpected).Path,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Manifest BSP source path does not match the selected platform"
}
$builtSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $builtSource).Hash.ToLowerInvariant()
if ($builtSourceHash -ne $patchedHash -or
    $builtSourceHash -ne ([string]$override.build_outputs.bsp_copied_source.sha256).ToLowerInvariant()) {
    throw "Vitis BSP copied source does not match the staged YT8521 patch"
}

$archiveExpected = Join-Path $workspacePath (
    "$platformName\export\$platformName\sw\standalone_psu_cortexa53_0\lib\liblwip220.a"
)
$archivePath = Resolve-WorkspaceFile `
    -Path ([string]$override.build_outputs.export_archive.path) `
    -Label "Exported lwip220 archive"
if (-not [string]::Equals(
    $archivePath,
    (Resolve-Path -LiteralPath $archiveExpected).Path,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Manifest lwip220 archive path does not match the selected platform"
}
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
if ($archiveHash -ne ([string]$override.build_outputs.export_archive.sha256).ToLowerInvariant()) {
    throw "Exported lwip220 archive SHA-256 mismatch"
}
$archiveBytes = [System.IO.File]::ReadAllBytes($archivePath)
if ($archiveBytes.Length -lt 8 -or
    [System.Text.Encoding]::ASCII.GetString($archiveBytes, 0, 8) -ne "!<arch>`n") {
    throw "Exported lwip220 output is not an archive"
}

$xsaPath = Resolve-WorkspaceFile `
    -Path ([string]$manifest.xsa_snapshot) `
    -Label "Network XSA snapshot"
$xsaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $xsaPath).Hash.ToLowerInvariant()
if ($xsaHash -ne ([string]$manifest.xsa_sha256).ToLowerInvariant()) {
    throw "Network XSA snapshot SHA-256 mismatch"
}

$bitCandidates = @(
    Get-ChildItem -LiteralPath (
        Join-Path $workspacePath "$platformName\export\$platformName\hw\sdt"
    ) `
        -Filter "*.bit" -File
)
if ($bitCandidates.Count -ne 1) {
    throw "Expected exactly one exported network bitstream, found $($bitCandidates.Count)"
}
$bitPath = $bitCandidates[0].FullName
$bitHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bitPath).Hash.ToLowerInvariant()
if ($bitHash -ne ([string]$manifest.bitstream_sha256).ToLowerInvariant()) {
    throw "Exported network bitstream SHA-256 mismatch"
}

$fsblPath = Resolve-WorkspaceFile `
    -Path (Join-Path $workspacePath "$platformName\export\$platformName\sw\boot\fsbl.elf") `
    -Label "Network FSBL"
$echoPath = Resolve-WorkspaceFile `
    -Path (Join-Path $workspacePath "$echoName\build\$echoName.elf") `
    -Label "Network echo ELF"
$manifestEchoPath = Resolve-WorkspaceFile `
    -Path ([string]$manifest.echo_application.elf.path) `
    -Label "Manifest network echo ELF"
if ($manifest.echo_application.name -ne $echoName -or
    -not [string]::Equals(
        $manifestEchoPath,
        $echoPath,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Manifest echo ELF does not match the selected application"
}
$echoHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $echoPath).Hash.ToLowerInvariant()
if ($echoHash -ne ([string]$manifest.echo_application.elf.sha256).ToLowerInvariant()) {
    throw "Network echo ELF SHA-256 mismatch"
}
$echoBytes = [System.IO.File]::ReadAllBytes($echoPath)
$echoAscii = [System.Text.Encoding]::ASCII.GetString($echoBytes)
$requiredEchoMarkers = @(
    "Detected Motorcomm YT8521",
    "YT8521 link resolved"
)
if (@(Compare-Object (
    @($manifest.echo_application.elf.required_patch_markers) | Sort-Object
) ($requiredEchoMarkers | Sort-Object)).Count -ne 0) {
    throw "Manifest echo ELF marker contract is incomplete"
}
foreach ($marker in $requiredEchoMarkers) {
    if (-not $echoAscii.Contains($marker)) {
        throw "Network echo ELF lacks required YT8521 marker '$marker'"
    }
}

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

$launcher = Join-Path $PSScriptRoot "launch_network_echo.tcl"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Network XSDB launcher is missing: $launcher"
}

$previous = @{
    QWEB_NETWORK_BIT = $env:QWEB_NETWORK_BIT
    QWEB_NETWORK_XSA = $env:QWEB_NETWORK_XSA
    QWEB_NETWORK_FSBL = $env:QWEB_NETWORK_FSBL
    QWEB_NETWORK_ECHO_ELF = $env:QWEB_NETWORK_ECHO_ELF
}
try {
    $env:QWEB_NETWORK_BIT = $bitPath
    $env:QWEB_NETWORK_XSA = $xsaPath
    $env:QWEB_NETWORK_FSBL = $fsblPath
    $env:QWEB_NETWORK_ECHO_ELF = $echoPath

    $xsdbCommand = $Xsdb
    $xsdbArguments = @($launcher)
    if ([System.IO.Path]::GetFileName($Xsdb) -ieq "xsdb.bat") {
        $xsdbCommand = Join-Path (Split-Path -Parent $Xsdb) "loader.bat"
        if (-not (Test-Path -LiteralPath $xsdbCommand -PathType Leaf)) {
            throw "AMD loader.bat was not found next to '$Xsdb'."
        }
        $xsdbArguments = @("-exec", "xsdb", $launcher)
    }

    & $xsdbCommand @xsdbArguments
    if ($LASTEXITCODE -ne 0) {
        throw "XSDB network launcher failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:QWEB_NETWORK_BIT = $previous.QWEB_NETWORK_BIT
    $env:QWEB_NETWORK_XSA = $previous.QWEB_NETWORK_XSA
    $env:QWEB_NETWORK_FSBL = $previous.QWEB_NETWORK_FSBL
    $env:QWEB_NETWORK_ECHO_ELF = $previous.QWEB_NETWORK_ECHO_ELF
}

Write-Host "PASS launched audited network echo image"
Write-Host "  bitstream SHA-256: $bitHash"
Write-Host "  XSA SHA-256: $xsaHash"
Write-Host "  echo ELF SHA-256: $echoHash"
