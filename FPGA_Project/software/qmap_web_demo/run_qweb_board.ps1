param(
    [string]$Workspace = "F:\vwk",
    [string]$RuntimeWorkbench = "F:\qot_boardtest_prompt_text_v13_20260812",
    [string]$Xsdb = "",
    [string]$EvidenceDirectory = "",
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# These hashes bind the launcher to the formal clean network build and the
# exact prompt workbench that passed the 2026-08-12 cold board acceptance.
$ExpectedNetworkManifestSha256 = "33ca1a825aff72dd7a59c9938c0b0e838062c5c6e18ee1826432341e7a85e401"
$ExpectedNetworkXsaSha256 = "0af1257442a68a6beb31d94811713f2ae8e6af63e0a85dd497146405e37406cf"
$ExpectedNetworkBitSha256 = "c926b3db8021e976e5ea6cc2f71da3c44899ea5c3df61da573475ce6d8c21239"
$ExpectedNetworkFsblSha256 = "a7695cd19bec264ca0ef1d951527587cc02489a82711ab63f4534c5f3579f5cb"
$ExpectedQwebElfSha256 = "38a772f093ce3996177640863888b6700f1ac26f7bcb82e5f2159c0ef46f89da"
$ExpectedWorkbenchManifestSha256 = "f9d523ab59926c7583f76e6faefc941d019b296ed5f51d00f49cf6627bc0eda7"
$ExpectedRuntimeManifestSha256 = "fa8981e71101def29970135df5e863da5634274dd9fb64905666f0cf1d47d3f2"
$ExpectedRuntimeLoaderSha256 = "0b698f28165f05efe7dbc2a8c64bfa7379cbf10cc8dcb4b1ced9660eeabbecb1"
$ExpectedRuntimeSegments = 61
$ExpectedRuntimeBytes = 394547200L
$ExpectedTokenizerBytes = 3629566L
$ExpectedTokenizerSha256 = "c20242603ef4144e3f3f2ec4ba97c0e9c315aadd41f1bd2c5740e2a7ffa03a7d"
$ExpectedXsdbSha256 = "2cd1d63586a95fbb00b2badb6652cae485bdd4cbdab612c8a5de0afd7e874f59"
$ExpectedLoaderSha256 = "13fac52e8f0c81ea0539a0edddf8ad839d5c8a49d9fe37531abffbab8decfb38"
$ExpectedSetupEnvSha256 = "0f8108b67a75b3957b075c3f87d094fcc63eb7251a27281e7d456024898d881f"
$ExpectedRdiArgsSha256 = "44f7765ce290ea0546f441044a5303316f7fe6c9434136ba013e8f72c5e1f544"
$ExpectedXsdbExeSha256 = "9ef5fe040728b508a1a0efd1dfd83dbb3df02d2d9b59310ea0e7b403cd429483"
$ExpectedXsdbManifestSha256 = "77596430e6ea9d0a8d7b686d75f0b554564fdc0a931d1f806cf16c0e100a99e3"
$ExpectedVitisRoot = "D:\Applications\Vivado_2025.1.1\2025.1.1\Vitis"

if (-not $AuditOnly -and [string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    throw "Physical QWEB launch requires -EvidenceDirectory from an active capture_qweb_uart.py session"
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    # Use the framework implementation directly.  This remains available when
    # a conda-launched PowerShell has a reduced PSModulePath and cannot
    # auto-load Microsoft.PowerShell.Utility/Get-FileHash.
    $stream = [System.IO.File]::OpenRead($Path)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $algorithm.ComputeHash($stream)
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
    return [System.BitConverter]::ToString($digest).Replace("-", "").ToLowerInvariant()
}

function Resolve-ExistingDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label is missing or is not a directory: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-ContainedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing or is not a regular file: $Path"
    }
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $prefix = $resolvedRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes its audited root: $resolved"
    }
    return $resolved
}

function Resolve-RelativeFile {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ([string]::IsNullOrWhiteSpace($RelativePath) -or
        [System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Label is not a safe relative path: $RelativePath"
    }
    $parts = @($RelativePath -split '[\\/]')
    if ($parts.Count -eq 0 -or @($parts | Where-Object {
        [string]::IsNullOrWhiteSpace($_) -or $_ -eq '.' -or $_ -eq '..'
    }).Count -ne 0) {
        throw "$Label contains an unsafe path component: $RelativePath"
    }
    return Resolve-ContainedFile -Root $Root `
        -Path (Join-Path $Root ($parts -join [System.IO.Path]::DirectorySeparatorChar)) `
        -Label $Label
}

function Assert-Hash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $actual = Get-Sha256 -Path $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch: got $actual expected $($Expected.ToLowerInvariant())"
    }
    return $actual
}

function Write-JsonExclusive {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )
    $temporary = "$Path.tmp"
    if ((Test-Path -LiteralPath $Path) -or (Test-Path -LiteralPath $temporary)) {
        throw "Refusing to overwrite QWEB launch evidence: $Path"
    }
    $json = ($Payload | ConvertTo-Json -Depth 8) + "`n"
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = $null
    $writer = $null
    try {
        $stream = [System.IO.File]::Open(
            $temporary,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $writer = New-Object System.IO.StreamWriter -ArgumentList $stream, $encoding
        $writer.Write($json)
        $writer.Flush()
        $stream.Flush($true)
    } finally {
        if ($null -ne $writer) {
            $writer.Dispose()
        } elseif ($null -ne $stream) {
            $stream.Dispose()
        }
    }
    [System.IO.File]::Move($temporary, $Path)
}

function Write-TextExclusive {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $temporary = "$Path.tmp"
    if ((Test-Path -LiteralPath $Path) -or (Test-Path -LiteralPath $temporary)) {
        throw "Refusing to overwrite QWEB launch evidence: $Path"
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $stream = $null
    $writer = $null
    try {
        $stream = [System.IO.File]::Open(
            $temporary,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $writer = New-Object System.IO.StreamWriter -ArgumentList $stream, $encoding
        $writer.Write($Text)
        $writer.Flush()
        $stream.Flush($true)
    } finally {
        if ($null -ne $writer) {
            $writer.Dispose()
        } elseif ($null -ne $stream) {
            $stream.Dispose()
        }
    }
    [System.IO.File]::Move($temporary, $Path)
}

function Assert-AArch64Elf {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 64
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length) {
            throw "$Label is shorter than one ELF64 header"
        }
    } finally {
        $stream.Dispose()
    }
    if ($header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
        $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
        $header[4] -ne 2 -or $header[5] -ne 1) {
        throw "$Label is not a little-endian ELF64 file"
    }
    $elfType = [System.BitConverter]::ToUInt16($header, 16)
    $machine = [System.BitConverter]::ToUInt16($header, 18)
    if ($elfType -ne 2 -or $machine -ne 183) {
        throw "$Label is not an AArch64 ET_EXEC ELF (type=$elfType machine=$machine)"
    }
}

function Assert-ExactStringSet {
    param(
        [Parameter(Mandatory = $true)][object[]]$Actual,
        [Parameter(Mandatory = $true)][string[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $actualStrings = @($Actual | ForEach-Object { [string]$_ })
    if ($actualStrings.Count -ne $Expected.Count -or
        @(Compare-Object ($actualStrings | Sort-Object) ($Expected | Sort-Object)).Count -ne 0) {
        throw "$Label does not match the pinned contract"
    }
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    try {
        return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    } catch {
        throw "$Label is not valid JSON: $($_.Exception.Message)"
    }
}

$workspacePath = Resolve-ExistingDirectory -Path $Workspace -Label "QWEB Vitis workspace"
$networkManifestPath = Resolve-ContainedFile -Root $workspacePath `
    -Path (Join-Path $workspacePath "network_workspace_manifest.json") `
    -Label "Network workspace manifest"
Assert-Hash -Path $networkManifestPath -Expected $ExpectedNetworkManifestSha256 `
    -Label "Network workspace manifest" | Out-Null
$networkManifest = Read-JsonFile -Path $networkManifestPath -Label "Network workspace manifest"

if (-not [string]::Equals([string]$networkManifest.workspace, $workspacePath,
    [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Network manifest workspace does not match the selected workspace"
}
if ([string]$networkManifest.platform_name -ne "p_net" -or
    [string]$networkManifest.app_name -ne "a_net_echo" -or
    [string]$networkManifest.template -ne "lwip_echo_server") {
    throw "Network manifest component/template identity does not match the pinned build"
}
if ([int]$networkManifest.build_results.platform -ne 0 -or
    [int]$networkManifest.build_results.application -ne 0 -or
    [int]$networkManifest.build_results.web_application -ne 0) {
    throw "Network manifest does not record three successful builds"
}

$expectedBspConfig = @(
    "lwip220|lwip220_api_mode|RAW_API",
    "lwip220|lwip220_dhcp|true",
    "lwip220|lwip220_ipv6_enable|false",
    "lwip220|lwip220_lwip_dhcp_does_acd_check|true",
    "lwip220|lwip220_memp_n_tcp_pcb|256",
    "lwip220|lwip220_pbuf_pool_size|2048",
    "xiltimer|XILTIMER_en_interval_timer|true"
)
$actualBspConfig = @($networkManifest.bsp_config | ForEach-Object {
    "{0}|{1}|{2}" -f $_.library, $_.parameter, $_.value
})
Assert-ExactStringSet -Actual $actualBspConfig -Expected $expectedBspConfig `
    -Label "Network BSP configuration"
Assert-ExactStringSet -Actual @($networkManifest.bsp_libraries) -Expected @("lwip220") `
    -Label "Network BSP library list"

$overrides = @($networkManifest.bsp_library_overrides)
if ($overrides.Count -ne 1) {
    throw "Network manifest must contain exactly one BSP override"
}
$override = $overrides[0]
$expectedPatchedHash = "4900358c786793496501e0a19d0970e43ad70e378d909ab2c8d7458cc1bac930"
if ([string]$override.library -ne "lwip220" -or
    [string]$override.version -ne "lwip220_v1_2" -or
    [string]$override.motorcomm_phy_id -ne "0x0000011A" -or
    ([string]$override.original_sha256).ToLowerInvariant() -ne
        "b45bad2d4c9e2543db7ec8e70b7b450633d748de749b8e67dc4f27894a63430d" -or
    ([string]$override.metadata_sha256).ToLowerInvariant() -ne
        "898d479f1ff6b828ab4666f371ca067e69b21b8b767d4b412422c18e8b09800e" -or
    ([string]$override.patched_sha256).ToLowerInvariant() -ne $expectedPatchedHash) {
    throw "Network manifest YT8521 lwip220 override does not match the pinned contract"
}
$overrideRoot = Resolve-ExistingDirectory -Path ([string]$override.destination) `
    -Label "Staged lwip220 override"
$workspacePrefix = $workspacePath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $overrideRoot.StartsWith($workspacePrefix,
    [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Staged lwip220 override escapes the audited workspace"
}
$patchedSource = Resolve-RelativeFile -Root $overrideRoot `
    -RelativePath ([string]$override.patched_file) -Label "Staged YT8521 source"
Assert-Hash -Path $patchedSource -Expected $expectedPatchedHash `
    -Label "Staged YT8521 source" | Out-Null

$builtSourceExpected = Join-Path $workspacePath (
    "p_net\psu_cortexa53_0\standalone_psu_cortexa53_0\bsp\libsrc\lwip220\" +
    ([string]$override.patched_file).Replace('/', '\')
)
$builtSource = Resolve-ContainedFile -Root $workspacePath -Path $builtSourceExpected `
    -Label "Vitis BSP copied YT8521 source"
if (-not [string]::Equals($builtSource,
    (Resolve-Path -LiteralPath ([string]$override.build_outputs.bsp_copied_source.path)).Path,
    [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Network manifest BSP copied source path is not the pinned platform output"
}
Assert-Hash -Path $builtSource -Expected $expectedPatchedHash `
    -Label "Vitis BSP copied YT8521 source" | Out-Null

$archivePath = Resolve-ContainedFile -Root $workspacePath `
    -Path (Join-Path $workspacePath "p_net\export\p_net\sw\standalone_psu_cortexa53_0\lib\liblwip220.a") `
    -Label "Exported lwip220 archive"
if (-not [string]::Equals($archivePath,
    (Resolve-Path -LiteralPath ([string]$override.build_outputs.export_archive.path)).Path,
    [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Network manifest lwip220 archive path is not the pinned platform output"
}
Assert-Hash -Path $archivePath `
    -Expected ([string]$override.build_outputs.export_archive.sha256) `
    -Label "Exported lwip220 archive" | Out-Null
$archiveHeader = New-Object byte[] 8
$archiveStream = [System.IO.File]::OpenRead($archivePath)
try {
    if ($archiveStream.Read($archiveHeader, 0, 8) -ne 8 -or
        [System.Text.Encoding]::ASCII.GetString($archiveHeader) -ne "!<arch>`n") {
        throw "Exported lwip220 output is not an archive"
    }
} finally {
    $archiveStream.Dispose()
}

$xsaPath = Resolve-ContainedFile -Root $workspacePath `
    -Path ([string]$networkManifest.xsa_snapshot) -Label "Network XSA snapshot"
if (([string]$networkManifest.xsa_sha256).ToLowerInvariant() -ne $ExpectedNetworkXsaSha256) {
    throw "Network manifest XSA hash is not the pinned formal-build hash"
}
Assert-Hash -Path $xsaPath -Expected $ExpectedNetworkXsaSha256 `
    -Label "Network XSA snapshot" | Out-Null
if ([System.IO.Path]::GetFileName($xsaPath) -ne
    "network_input_$ExpectedNetworkXsaSha256.xsa") {
    throw "Network XSA snapshot is not content-addressed by its pinned hash"
}

$bitCandidates = @(Get-ChildItem -LiteralPath (Join-Path $workspacePath "p_net\export\p_net\hw\sdt") `
    -Filter "*.bit" -File)
if ($bitCandidates.Count -ne 1) {
    throw "Expected exactly one exported network bitstream, found $($bitCandidates.Count)"
}
$bitPath = Resolve-ContainedFile -Root $workspacePath -Path $bitCandidates[0].FullName `
    -Label "Network bitstream"
if (([string]$networkManifest.bitstream_sha256).ToLowerInvariant() -ne $ExpectedNetworkBitSha256) {
    throw "Network manifest bitstream hash is not the pinned formal-build hash"
}
Assert-Hash -Path $bitPath -Expected $ExpectedNetworkBitSha256 `
    -Label "Network bitstream" | Out-Null

$fsblPath = Resolve-ContainedFile -Root $workspacePath `
    -Path (Join-Path $workspacePath "p_net\export\p_net\sw\boot\fsbl.elf") `
    -Label "Network FSBL"
Assert-Hash -Path $fsblPath -Expected $ExpectedNetworkFsblSha256 `
    -Label "Network FSBL" | Out-Null
Assert-AArch64Elf -Path $fsblPath -Label "Network FSBL"

$web = $networkManifest.web_application
if ([string]$web.name -ne "a_qweb" -or [string]$web.template -ne "lwip_echo_server" -or
    [int64]$web.stack_bytes -ne 65536 -or [int64]$web.heap_bytes -ne 65536 -or
    [int64]$web.elf.tokenizer_bytes -ne $ExpectedTokenizerBytes -or
    ([string]$web.tokenizer_sha256).ToLowerInvariant() -ne $ExpectedTokenizerSha256) {
    throw "Network manifest Web application contract does not match a_qweb"
}
$expectedCompileSources = @(
    "echo.c", "i2c_access.c", "iic_phyreset.c", "main.c", "platform.c",
    "platform_mb.c", "platform_zynq.c", "platform_zynqmp.c", "sfp.c",
    "si5324.c", "qot_session.c", "qtk_tokenizer_runtime.c",
    "qtk_text_tokenizer.c", "qweb_http.c", "qweb_api.c", "qweb_job.c",
    "qweb_router.c", "qweb_lwip_adapter.c", "qweb_board_app.c",
    "web_assets.c", "tokenizer_asset.S"
)
Assert-ExactStringSet -Actual @($web.compile_sources) -Expected $expectedCompileSources `
    -Label "a_qweb compile source list"

$inventory = @($web.staged_inventory)
if ($inventory.Count -ne 51) {
    throw "a_qweb staged inventory count mismatch: $($inventory.Count)"
}
$inventoryNames = New-Object 'System.Collections.Generic.HashSet[string]' `
    ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $inventory) {
    $relative = [string]$entry.path
    if (-not $inventoryNames.Add($relative)) {
        throw "a_qweb staged inventory contains duplicate path: $relative"
    }
    $inventoryFile = Resolve-RelativeFile -Root (Join-Path $workspacePath "a_qweb") `
        -RelativePath $relative -Label "a_qweb staged inventory file"
    $actualBytes = (Get-Item -LiteralPath $inventoryFile).Length
    if ($actualBytes -ne [int64]$entry.bytes) {
        throw "a_qweb staged file size mismatch: $relative"
    }
    Assert-Hash -Path $inventoryFile -Expected ([string]$entry.sha256) `
        -Label "a_qweb staged file '$relative'" | Out-Null
}

$qwebPath = Resolve-ContainedFile -Root $workspacePath `
    -Path (Join-Path $workspacePath "a_qweb\build\a_qweb.elf") -Label "a_qweb ELF"
$manifestQwebPath = Resolve-ContainedFile -Root $workspacePath `
    -Path ([string]$web.elf.path) -Label "Manifest a_qweb ELF"
if (-not [string]::Equals($qwebPath, $manifestQwebPath,
    [System.StringComparison]::OrdinalIgnoreCase) -or
    ([string]$web.elf.sha256).ToLowerInvariant() -ne $ExpectedQwebElfSha256 -or
    (Get-Item -LiteralPath $qwebPath).Length -ne [int64]$web.elf.bytes) {
    throw "a_qweb ELF path/size/hash metadata does not match the pinned build"
}
Assert-Hash -Path $qwebPath -Expected $ExpectedQwebElfSha256 -Label "a_qweb ELF" | Out-Null
Assert-AArch64Elf -Path $qwebPath -Label "a_qweb ELF"
$requiredSymbols = @(
    "main", "start_application", "transfer_data", "qweb_board_qot_runner",
    "qweb_job_step", "qot_session_step", "qtk_tokenize_utf8"
)
Assert-ExactStringSet -Actual @($web.elf.required_symbols) -Expected $requiredSymbols `
    -Label "a_qweb required symbol list"
$qwebAscii = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($qwebPath))
foreach ($symbol in $requiredSymbols) {
    if (-not $qwebAscii.Contains($symbol)) {
        throw "a_qweb ELF lacks required symbol string '$symbol'"
    }
}
$requiredQwebMarkers = @(
    "Detected Motorcomm YT8521",
    "YT8521 link resolved",
    "QWEB READY http://",
    "TOKENIZER tokens="
)
foreach ($marker in $requiredQwebMarkers) {
    if (-not $qwebAscii.Contains($marker)) {
        throw "a_qweb ELF lacks required linked marker '$marker'"
    }
}

$runtimeRoot = Resolve-ExistingDirectory -Path $RuntimeWorkbench `
    -Label "Board-accepted runtime workbench"
$workbenchManifestPath = Resolve-ContainedFile -Root $runtimeRoot `
    -Path (Join-Path $runtimeRoot "workbench_manifest.json") `
    -Label "Runtime workbench manifest"
Assert-Hash -Path $workbenchManifestPath -Expected $ExpectedWorkbenchManifestSha256 `
    -Label "Runtime workbench manifest" | Out-Null
$workbench = Read-JsonFile -Path $workbenchManifestPath -Label "Runtime workbench manifest"
if ([int]$workbench.format_version -ne 5 -or
    [string]$workbench.state -ne "WORKBENCH_NOT_RELEASE" -or
    [string]$workbench.legacy_lineage_id -ne "full28-board-pass-20260808" -or
    [int64]$workbench.runtime.segment_count -ne $ExpectedRuntimeSegments -or
    [int64]$workbench.runtime.total_segment_bytes -ne $ExpectedRuntimeBytes -or
    [string]$workbench.runtime.download_command -ne "dow -data -bypass-cache-sync" -or
    ([string]$workbench.runtime.manifest_sha256).ToLowerInvariant() -ne
        $ExpectedRuntimeManifestSha256 -or
    ([string]$workbench.runtime.loader_sha256).ToLowerInvariant() -ne
        $ExpectedRuntimeLoaderSha256) {
    throw "Runtime workbench manifest does not match the board-accepted v13 contract"
}
if (@($workbench.files).Count -ne [int]$workbench.file_count -or
    (@($workbench.files | Measure-Object -Property nbytes -Sum).Sum) -ne
        [int64]$workbench.total_file_bytes) {
    throw "Runtime workbench inventory count/byte totals are inconsistent"
}
$workbenchFiles = @{}
foreach ($entry in @($workbench.files)) {
    $name = ([string]$entry.path).Replace('\', '/')
    if ($workbenchFiles.ContainsKey($name)) {
        throw "Runtime workbench manifest contains duplicate path: $name"
    }
    $workbenchFiles[$name] = $entry
}

$runtimeDirectory = Resolve-ExistingDirectory -Path (Join-Path $runtimeRoot "runtime") `
    -Label "Board-accepted runtime directory"
$runtimeLoader = Resolve-ContainedFile -Root $runtimeRoot `
    -Path (Join-Path $runtimeDirectory "load_pl_ddr_runtime.tcl") `
    -Label "Qwen runtime loader"
$segmentManifestPath = Resolve-ContainedFile -Root $runtimeRoot `
    -Path (Join-Path $runtimeDirectory "pl_ddr_binary_segments.json") `
    -Label "Qwen segment manifest"
Assert-Hash -Path $runtimeLoader -Expected $ExpectedRuntimeLoaderSha256 `
    -Label "Qwen runtime loader" | Out-Null
Assert-Hash -Path $segmentManifestPath -Expected $ExpectedRuntimeManifestSha256 `
    -Label "Qwen segment manifest" | Out-Null
foreach ($supportPath in @("runtime/full_chain_manifest.json", "runtime/pl_ddr_runtime_load_plan.json")) {
    if (-not $workbenchFiles.ContainsKey($supportPath)) {
        throw "Runtime workbench inventory lacks $supportPath"
    }
    $supportFile = Resolve-RelativeFile -Root $runtimeRoot -RelativePath $supportPath `
        -Label "Runtime support manifest"
    Assert-Hash -Path $supportFile -Expected ([string]$workbenchFiles[$supportPath].sha256) `
        -Label $supportPath | Out-Null
}

$segmentManifest = Read-JsonFile -Path $segmentManifestPath -Label "Qwen segment manifest"
$segments = @($segmentManifest.segments)
if ([int]$segmentManifest.format_version -ne 1 -or
    [int64]$segmentManifest.segment_count -ne $ExpectedRuntimeSegments -or
    [int64]$segmentManifest.total_segment_bytes -ne $ExpectedRuntimeBytes -or
    $segments.Count -ne $ExpectedRuntimeSegments) {
    throw "Qwen segment manifest count/byte contract is invalid"
}
$runtimeFiles = @(Get-ChildItem -LiteralPath $runtimeDirectory -File -Filter "qwen3_runtime_*.bin")
if ($runtimeFiles.Count -ne $ExpectedRuntimeSegments) {
    throw "Runtime directory must contain exactly $ExpectedRuntimeSegments segment files"
}

$apertureStart = [Convert]::ToUInt64("400000000", 16)
$apertureLimit = [Convert]::ToUInt64("420000000", 16)
$intervals = @()
$totalRuntimeBytes = 0L
$loaderText = Get-Content -Raw -LiteralPath $runtimeLoader
$loaderSetCount = ([regex]::Matches($loaderText,
    'set segment_file \[file join \$script_dir "qwen3_runtime_[0-9]{2}\.bin"\]')).Count
$loaderDownloadCount = ([regex]::Matches($loaderText,
    '(?m)^dow -data -bypass-cache-sync \$segment_file 0x[0-9A-F]{16}\r?$')).Count
if ($loaderSetCount -ne $ExpectedRuntimeSegments -or
    $loaderDownloadCount -ne $ExpectedRuntimeSegments) {
    throw "Qwen runtime loader does not contain exactly 61 audited download commands"
}

for ($index = 0; $index -lt $segments.Count; $index++) {
    $segment = $segments[$index]
    $expectedName = "qwen3_runtime_{0:D2}.bin" -f $index
    if ([int]$segment.index -ne $index -or [string]$segment.file -ne $expectedName) {
        throw "Qwen segment ordering/name mismatch at index $index"
    }
    $address = [uint64]$segment.address
    $nbytes = [uint64]$segment.nbytes
    $limit = $address + $nbytes
    if ($nbytes -eq 0 -or ($address % 4) -ne 0 -or ($nbytes % 4) -ne 0 -or
        $address -lt $apertureStart -or $limit -gt $apertureLimit -or $limit -le $address) {
        throw "Qwen segment $expectedName is outside/aligned incorrectly for PL DDR"
    }
    $expectedAddressHex = "0x{0:X16}" -f $address
    if ([string]$segment.address_hex -cne $expectedAddressHex) {
        throw "Qwen segment $expectedName address_hex mismatch"
    }
    $relativeName = "runtime/$expectedName"
    if (-not $workbenchFiles.ContainsKey($relativeName)) {
        throw "Runtime workbench inventory lacks $relativeName"
    }
    $segmentPath = Resolve-ContainedFile -Root $runtimeRoot `
        -Path (Join-Path $runtimeDirectory $expectedName) -Label "Qwen runtime segment"
    if ((Get-Item -LiteralPath $segmentPath).Length -ne [int64]$nbytes -or
        [int64]$workbenchFiles[$relativeName].nbytes -ne [int64]$nbytes -or
        ([string]$workbenchFiles[$relativeName].sha256).ToLowerInvariant() -ne
            ([string]$segment.sha256).ToLowerInvariant()) {
        throw "Qwen segment $expectedName size/inventory metadata mismatch"
    }
    Assert-Hash -Path $segmentPath -Expected ([string]$segment.sha256) `
        -Label "Qwen segment $expectedName" | Out-Null
    $setLine = 'set segment_file [file join $script_dir "' + $expectedName + '"]'
    $downloadLine = 'dow -data -bypass-cache-sync $segment_file ' + $expectedAddressHex
    if (-not $loaderText.Contains($setLine) -or -not $loaderText.Contains($downloadLine)) {
        throw "Qwen runtime loader lacks exact command for $expectedName"
    }
    $intervals += [pscustomobject]@{ Start = $address; Limit = $limit; Name = $expectedName }
    $totalRuntimeBytes += [int64]$nbytes
}
if ($totalRuntimeBytes -ne $ExpectedRuntimeBytes) {
    throw "Qwen runtime segment byte sum mismatch: $totalRuntimeBytes"
}
$sortedIntervals = @($intervals | Sort-Object Start)
for ($index = 1; $index -lt $sortedIntervals.Count; $index++) {
    if ([uint64]$sortedIntervals[$index].Start -lt [uint64]$sortedIntervals[$index - 1].Limit) {
        throw "Qwen runtime segments overlap: $($sortedIntervals[$index - 1].Name) and $($sortedIntervals[$index].Name)"
    }
}

Write-Host "PASS audited board-hosted QWEB launch inputs"
Write-Host "  network manifest SHA-256: $ExpectedNetworkManifestSha256"
Write-Host "  network bitstream SHA-256: $ExpectedNetworkBitSha256"
Write-Host "  network XSA SHA-256: $ExpectedNetworkXsaSha256"
Write-Host "  network FSBL SHA-256: $ExpectedNetworkFsblSha256"
Write-Host "  a_qweb ELF SHA-256: $ExpectedQwebElfSha256"
Write-Host "  runtime: $ExpectedRuntimeSegments segments / $ExpectedRuntimeBytes bytes"
Write-Host "  runtime manifest SHA-256: $ExpectedRuntimeManifestSha256"

if ($AuditOnly) {
    Write-Host "PASS audit-only mode; XSDB was not invoked"
    return
}

$evidencePath = Resolve-ExistingDirectory -Path $EvidenceDirectory `
    -Label "QWEB evidence directory"
$evidenceRawPath = Join-Path $evidencePath "uart_raw.bin"
$evidenceStartupPath = Join-Path $evidencePath "startup.json"
$launchReportPath = Join-Path $evidencePath "launch.json"
$launchClaimPath = Join-Path $evidencePath "launch.claim.json"
$xsdbLogPath = Join-Path $evidencePath "xsdb.log"
$xsdbProfilePath = Join-Path $evidencePath "xsdb_profile"
if (-not (Test-Path -LiteralPath $evidenceRawPath -PathType Leaf)) {
    throw "QWEB evidence directory is not owned by an active UART capture: missing $evidenceRawPath"
}
if ((Test-Path -LiteralPath $evidenceStartupPath) -or
    (Test-Path -LiteralPath $launchReportPath) -or
    (Test-Path -LiteralPath "$launchReportPath.tmp") -or
    (Test-Path -LiteralPath $launchClaimPath) -or
    (Test-Path -LiteralPath "$launchClaimPath.tmp") -or
    (Test-Path -LiteralPath $xsdbLogPath) -or
    (Test-Path -LiteralPath "$xsdbLogPath.tmp") -or
    (Test-Path -LiteralPath $xsdbProfilePath)) {
    throw "QWEB evidence directory already contains completed or prior-run evidence: $evidencePath"
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
$xsdbPath = (Resolve-Path -LiteralPath $Xsdb).Path
$expectedVitisRootPath = (Resolve-Path -LiteralPath $ExpectedVitisRoot).Path
$expectedXsdbPath = Join-Path $expectedVitisRootPath "bin\xsdb.bat"
if (-not [string]::Equals(
        $xsdbPath,
        $expectedXsdbPath,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Physical QWEB launch requires canonical Vitis 2025.1 xsdb.bat at '$expectedXsdbPath', not '$xsdbPath'"
}
Assert-Hash -Path $xsdbPath -Expected $ExpectedXsdbSha256 `
    -Label "AMD xsdb.bat" | Out-Null
$loaderPath = Join-Path (Split-Path -Parent $xsdbPath) "loader.bat"
$setupEnvPath = Join-Path (Split-Path -Parent $xsdbPath) "setupEnv.bat"
$rdiArgsPath = Join-Path (Split-Path -Parent $xsdbPath) "rdiArgs.bat"
$xsdbExePath = Join-Path $expectedVitisRootPath "bin\unwrapped\win64.o\xsdb.exe"
$xsdbManifestPath = "$xsdbExePath.manifest"
$toolFiles = @(
    [pscustomobject]@{ Path = $loaderPath; Sha = $ExpectedLoaderSha256; Label = "AMD loader.bat" },
    [pscustomobject]@{ Path = $setupEnvPath; Sha = $ExpectedSetupEnvSha256; Label = "AMD setupEnv.bat" },
    [pscustomobject]@{ Path = $rdiArgsPath; Sha = $ExpectedRdiArgsSha256; Label = "AMD rdiArgs.bat" },
    [pscustomobject]@{ Path = $xsdbExePath; Sha = $ExpectedXsdbExeSha256; Label = "AMD xsdb.exe" },
    [pscustomobject]@{ Path = $xsdbManifestPath; Sha = $ExpectedXsdbManifestSha256; Label = "AMD xsdb.exe.manifest" }
)
foreach ($toolFile in $toolFiles) {
    if (-not (Test-Path -LiteralPath $toolFile.Path -PathType Leaf)) {
        throw "$($toolFile.Label) is missing: $($toolFile.Path)"
    }
    Assert-Hash -Path $toolFile.Path -Expected $toolFile.Sha `
        -Label $toolFile.Label | Out-Null
}
$baselinePath = Join-Path $expectedVitisRootPath "data\baseline.txt"
if (Test-Path -LiteralPath $baselinePath) {
    throw "Refusing Vitis baseline redirection during physical QWEB launch: $baselinePath"
}
$patchRoot = Join-Path $expectedVitisRootPath "patches"
if (Test-Path -LiteralPath $patchRoot -PathType Container) {
    $patchedXsdb = @(Get-ChildItem -LiteralPath $patchRoot -Filter "xsdb.exe" `
        -File -Recurse -ErrorAction Stop)
    if ($patchedXsdb.Count -ne 0) {
        throw "Refusing patched XSDB executable during physical QWEB launch: $($patchedXsdb[0].FullName)"
    }
}

$launcher = Join-Path $PSScriptRoot "launch_qweb_board.tcl"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "QWEB XSDB launcher is missing: $launcher"
}
$wrapperSha256 = Get-Sha256 -Path $PSCommandPath
$launcherSha256 = Get-Sha256 -Path $launcher

$launchRunId = [System.Guid]::NewGuid().ToString("D")
$launchStartedUtc = [System.DateTimeOffset]::UtcNow.ToString("o")
$launchClaim = [ordered]@{
    schema_version = 1
    tool = "run_qweb_board.ps1"
    state = "claimed"
    run_id = $launchRunId
    started_utc = $launchStartedUtc
    evidence_directory = $evidencePath
}
Write-JsonExclusive -Path $launchClaimPath -Payload $launchClaim
$launchClaimSha256 = Get-Sha256 -Path $launchClaimPath
[void][System.IO.Directory]::CreateDirectory($xsdbProfilePath)
if (@(Get-ChildItem -LiteralPath $xsdbProfilePath -Force).Count -ne 0) {
    throw "Isolated XSDB profile was not empty at creation: $xsdbProfilePath"
}

$previous = @{
    QWEB_NETWORK_BIT = $env:QWEB_NETWORK_BIT
    QWEB_NETWORK_XSA = $env:QWEB_NETWORK_XSA
    QWEB_NETWORK_FSBL = $env:QWEB_NETWORK_FSBL
    QWEB_NETWORK_WEB_ELF = $env:QWEB_NETWORK_WEB_ELF
    QWEB_RUNTIME_LOADER = $env:QWEB_RUNTIME_LOADER
}
$unsafeXsdbEnvironmentNames = @(
    "PATH",
    "PYTHONPATH",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOMESHARE",
    "USERPROFILE",
    "MYXILINX",
    "MYVIVADO",
    "XILINX",
    "XILINX_PATH",
    "XIL_NO_OVERRIDE",
    "XIL_PA_NO_DEFAULT_OVERRIDE",
    "XIL_PA_NO_XILINX_OVERRIDE",
    "XIL_PA_NO_XILINX_SDK_OVERRIDE",
    "XIL_PA_NO_XILINX_PATH_OVERRIDE",
    "XILINX_VITIS",
    "XILINX_VIVADO",
    "XILINX_SDK",
    "XILINX_HLS",
    "XILINX_VCXX",
    "XILINX_COMMON_TOOLS",
    "XIL_TPS_ROOT",
    "RDI_PATCHROOT",
    "_RDI_BASELINE",
    "_RDI_NEEDS_PYTHON",
    "RDI_BASELINE",
    "RDI_PREPEND_PATH",
    "RDI_MIXED_EXT",
    "RDI_DEPENDENCY",
    "RDI_BYPASS_ARGS",
    "RDI_PROG",
    "RDI_ARGS",
    "RDI_ARGS_FUNCTION",
    "RDI_SETUP_ENV_FUNCTION",
    "RDI_JAVALAUNCH",
    "RDI_VBSLAUNCH",
    "RDI_EXECCLASS",
    "RDI_CLASSPATH",
    "RDI_JAVAARGS",
    "RDI_JAVAFXROOT",
    "RDI_JAVACEFROOT",
    "RDI_APPROOT",
    "RDI_BINROOT",
    "RDI_INSTALLROOT",
    "RDI_PLATFORM",
    "RDI_OPT_EXT",
    "_RDI_SETENV_RUN",
    "QWEB_HW_SERVER_URL",
    "QWEB_DEVICE_FILTER",
    "TCLLIBPATH",
    "TCL_LIBRARY",
    "TK_LIBRARY"
)
$previousXsdbEnvironment = @{}
foreach ($name in $unsafeXsdbEnvironmentNames) {
    $previousXsdbEnvironment[$name] = `
        [System.Environment]::GetEnvironmentVariable($name, "Process")
}
$launchFailure = $null
$xsdbExitCode = $null
$xsdbLogSha256 = $null
$requiredXsdbMarkers = @(
    "PASS programmed audited network bitstream",
    "PASS FSBL initialization",
    "PASS PL DDR4 ready status=0x00000005",
    "PASS all 281 QMAP packet headers",
    "PASS a_qweb downloaded and running after audited runtime load"
)
$xsdbOutput = New-Object System.Collections.Generic.List[string]
$xsdbCommand = $xsdbPath
$xsdbArguments = @("-no-ini", $launcher)
$locationPushed = $false
try {
    foreach ($name in $unsafeXsdbEnvironmentNames) {
        [System.Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
    $env:HOME = $xsdbProfilePath
    $env:USERPROFILE = $xsdbProfilePath
    $env:QWEB_NETWORK_BIT = $bitPath
    $env:QWEB_NETWORK_XSA = $xsaPath
    $env:QWEB_NETWORK_FSBL = $fsblPath
    $env:QWEB_NETWORK_WEB_ELF = $qwebPath
    $env:QWEB_RUNTIME_LOADER = $runtimeLoader

    Push-Location -LiteralPath $xsdbProfilePath
    $locationPushed = $true
    & $xsdbCommand @xsdbArguments 2>&1 | ForEach-Object {
        $line = [string]$_
        [void]$xsdbOutput.Add($line)
        Write-Host $line
    }
    $xsdbExitCode = $LASTEXITCODE
    $xsdbText = ($xsdbOutput -join "`n") + "`n"
    Write-TextExclusive -Path $xsdbLogPath -Text $xsdbText
    $xsdbLogSha256 = Get-Sha256 -Path $xsdbLogPath
    if ($xsdbExitCode -ne 0) {
        throw "XSDB QWEB launcher failed with exit code $xsdbExitCode"
    }
    $markerPosition = -1
    foreach ($marker in $requiredXsdbMarkers) {
        $nextPosition = $xsdbText.IndexOf(
            $marker,
            $markerPosition + 1,
            [System.StringComparison]::Ordinal
        )
        if ($nextPosition -lt 0) {
            throw "XSDB QWEB launcher output lacks ordered PASS marker: $marker"
        }
        $markerPosition = $nextPosition
    }
} catch {
    $launchFailure = $_.Exception.Message
} finally {
    if ($locationPushed) {
        Pop-Location
    }
    $env:QWEB_NETWORK_BIT = $previous.QWEB_NETWORK_BIT
    $env:QWEB_NETWORK_XSA = $previous.QWEB_NETWORK_XSA
    $env:QWEB_NETWORK_FSBL = $previous.QWEB_NETWORK_FSBL
    $env:QWEB_NETWORK_WEB_ELF = $previous.QWEB_NETWORK_WEB_ELF
    $env:QWEB_RUNTIME_LOADER = $previous.QWEB_RUNTIME_LOADER
    foreach ($name in $unsafeXsdbEnvironmentNames) {
        [System.Environment]::SetEnvironmentVariable(
            $name,
            $previousXsdbEnvironment[$name],
            "Process"
        )
    }
}

$launchFinishedUtc = [System.DateTimeOffset]::UtcNow.ToString("o")
$launchReport = [ordered]@{
    schema_version = 1
    tool = "run_qweb_board.ps1"
    run_id = $launchRunId
    passed = ($null -eq $launchFailure)
    failure = $launchFailure
    started_utc = $launchStartedUtc
    finished_utc = $launchFinishedUtc
    evidence_directory = $evidencePath
    xsdb_exit_code = $xsdbExitCode
    claim = [ordered]@{
        path = "launch.claim.json"
        sha256 = $launchClaimSha256
    }
    audited = [ordered]@{
        network_manifest_sha256 = $ExpectedNetworkManifestSha256
        network_bit_sha256 = $ExpectedNetworkBitSha256
        network_xsa_sha256 = $ExpectedNetworkXsaSha256
        network_fsbl_sha256 = $ExpectedNetworkFsblSha256
        qweb_elf_sha256 = $ExpectedQwebElfSha256
        workbench_manifest_sha256 = $ExpectedWorkbenchManifestSha256
        runtime_manifest_sha256 = $ExpectedRuntimeManifestSha256
        runtime_loader_sha256 = $ExpectedRuntimeLoaderSha256
        runtime_segments = $ExpectedRuntimeSegments
        runtime_bytes = $ExpectedRuntimeBytes
        tokenizer_sha256 = $ExpectedTokenizerSha256
        tokenizer_bytes = $ExpectedTokenizerBytes
        xsdb_sha256 = $ExpectedXsdbSha256
        loader_sha256 = $ExpectedLoaderSha256
        setup_env_sha256 = $ExpectedSetupEnvSha256
        rdi_args_sha256 = $ExpectedRdiArgsSha256
        xsdb_exe_sha256 = $ExpectedXsdbExeSha256
        xsdb_manifest_sha256 = $ExpectedXsdbManifestSha256
        vitis_root = $expectedVitisRootPath
    }
    launcher = [ordered]@{
        wrapper = $PSCommandPath
        wrapper_sha256 = $wrapperSha256
        tcl = $launcher
        tcl_sha256 = $launcherSha256
        xsdb = $xsdbPath
        xsdb_sha256 = $ExpectedXsdbSha256
        loader = $loaderPath
        loader_sha256 = $ExpectedLoaderSha256
        setup_env = $setupEnvPath
        setup_env_sha256 = $ExpectedSetupEnvSha256
        rdi_args = $rdiArgsPath
        rdi_args_sha256 = $ExpectedRdiArgsSha256
        xsdb_exe = $xsdbExePath
        xsdb_exe_sha256 = $ExpectedXsdbExeSha256
        xsdb_manifest = $xsdbManifestPath
        xsdb_manifest_sha256 = $ExpectedXsdbManifestSha256
        command = $xsdbCommand
        arguments = $xsdbArguments
        working_directory = "xsdb_profile"
        isolated_home = "xsdb_profile"
        profile_initially_empty = $true
        sanitized_environment = $unsafeXsdbEnvironmentNames
        required_output_markers = $requiredXsdbMarkers
        output_log = "xsdb.log"
        output_log_sha256 = $xsdbLogSha256
    }
}
Write-JsonExclusive -Path $launchReportPath -Payload $launchReport
Write-Host "QWEB launch report: $launchReportPath"
Write-Host "QWEB launch run ID: $launchRunId"
if ($null -ne $launchFailure) {
    throw $launchFailure
}

Write-Host "PASS launched audited board-hosted QWEB image after runtime load"
