param(
    [string]$Xsdb = "",
    [string]$EvidenceDirectory = "",
    [string]$HwServerUrl = "tcp:127.0.0.1:3121",
    [string]$DeviceFilter = 'name =~ "PL"',
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
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

function Write-TextExclusive {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $temporary = "$Path.tmp"
    if ((Test-Path -LiteralPath $Path) -or (Test-Path -LiteralPath $temporary)) {
        throw "Refusing to overwrite launch evidence: $Path"
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

function Write-JsonExclusive {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Payload
    )
    $json = ($Payload | ConvertTo-Json -Depth 8) + "`n"
    Write-TextExclusive -Path $Path -Text $json
}

function Resolve-ContainedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Label must use a release-relative path: $RelativePath"
    }
    $candidate = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Label is missing: $candidate"
    }
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path.TrimEnd('\', '/')
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $prefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes the release directory: $resolved"
    }
    return $resolved
}

function Resolve-Xsdb {
    param(
        [string]$Requested,
        [Parameter(Mandatory = $true)][string]$DefaultPath
    )
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        if (-not (Test-Path -LiteralPath $Requested -PathType Leaf)) {
            throw "Explicit -Xsdb path does not exist: $Requested"
        }
        return (Resolve-Path -LiteralPath $Requested).Path
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($env:XILINX_VITIS)) {
        $candidates.Add((Join-Path $env:XILINX_VITIS "bin\xsdb.bat"))
    }
    $candidates.Add($DefaultPath)
    foreach ($name in @("xsdb.bat", "xsdb")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $candidates.Add($command.Source)
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "XSDB was not found. Install AMD Vitis 2025.1 or pass -Xsdb <path-to-xsdb.bat>."
}

function Assert-CaptureActive {
    param(
        [Parameter(Mandatory = $true)][string]$HeartbeatPath,
        [Parameter(Mandatory = $true)][string]$EvidenceRoot,
        [Parameter(Mandatory = $true)][Guid]$CaptureId,
        [Parameter(Mandatory = $true)][int]$CapturePid,
        [Parameter(Mandatory = $true)][DateTimeOffset]$CaptureStarted
    )
    if (-not (Test-Path -LiteralPath $HeartbeatPath -PathType Leaf)) {
        throw "UART capture heartbeat is missing: $HeartbeatPath"
    }
    $heartbeat = Get-Content -LiteralPath $HeartbeatPath -Raw | ConvertFrom-Json
    if ([int]$heartbeat.schema_version -ne 1 -or
        [string]$heartbeat.tool -ne "capture_qweb_uart.py" -or
        [string]$heartbeat.state -ne "capturing" -or
        [string]$heartbeat.capture_id -ne $CaptureId.ToString("D") -or
        [int]$heartbeat.pid -ne $CapturePid -or
        [long]$heartbeat.sequence -lt 1 -or
        -not [string]::Equals(
            (Resolve-Path -LiteralPath ([string]$heartbeat.evidence_directory)).Path,
            $EvidenceRoot,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "UART capture heartbeat is malformed, closed, or belongs to another capture"
    }
    $heartbeatUtc = [DateTimeOffset]::Parse([string]$heartbeat.heartbeat_utc)
    $now = [DateTimeOffset]::UtcNow
    if ($heartbeatUtc -lt $CaptureStarted -or
        $heartbeatUtc -gt $now.AddSeconds(1) -or
        ($now - $heartbeatUtc).TotalSeconds -gt 3) {
        throw "UART capture heartbeat is stale or has an invalid timestamp: $heartbeatUtc"
    }
    $captureProcess = Get-Process -Id $CapturePid -ErrorAction SilentlyContinue
    if ($null -eq $captureProcess) {
        throw "UART capture process $CapturePid is no longer running"
    }
    $processStarted = [DateTimeOffset]($captureProcess.StartTime.ToUniversalTime())
    if ($processStarted -gt $CaptureStarted.AddSeconds(1)) {
        throw "UART capture PID $CapturePid was reused by another process"
    }
    return [pscustomobject]@{
        heartbeat_utc = $heartbeatUtc.ToString("o")
        sequence = [long]$heartbeat.sequence
    }
}

$releaseRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$manifestPath = Join-Path $releaseRoot "release_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Release manifest is missing: $manifestPath"
}
$release = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ([int]$release.schema_version -ne 1) {
    throw "Unsupported release manifest schema: $($release.schema_version)"
}

$resolvedArtifacts = @{}
foreach ($artifact in $release.artifacts) {
    $path = Resolve-ContainedFile -Root $releaseRoot -RelativePath $artifact.path -Label $artifact.id
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -ne [long]$artifact.bytes) {
        throw "$($artifact.id) size mismatch: $size != $($artifact.bytes)"
    }
    $digest = Get-Sha256 -Path $path
    if ($digest -ne [string]$artifact.sha256) {
        throw "$($artifact.id) SHA-256 mismatch: $digest"
    }
    $resolvedArtifacts[$artifact.id] = $path
    Write-Host "PASS artifact $($artifact.id) $size bytes"
}

$runtime = $release.runtime
$segmentManifest = Get-Content -LiteralPath $resolvedArtifacts.segment_manifest -Raw | ConvertFrom-Json
$segments = @($segmentManifest.segments)
$expectedCount = [int]$runtime.expected_segment_count
$expectedBytes = [long]$runtime.expected_total_bytes
if ($segments.Count -ne $expectedCount -or [int]$segmentManifest.segment_count -ne $expectedCount) {
    throw "Runtime manifest does not contain exactly $expectedCount segments"
}
if ([long]$segmentManifest.total_segment_bytes -ne $expectedBytes) {
    throw "Runtime manifest byte total mismatch"
}

$modelRoot = Split-Path -Parent $resolvedArtifacts.segment_manifest
$expectedNames = @()
for ($index = 0; $index -lt $expectedCount; $index++) {
    $expectedNames += ("qwen3_runtime_{0:D2}.bin" -f $index)
}
$actualNames = @(Get-ChildItem -LiteralPath $modelRoot -Filter "qwen3_runtime_*.bin" -File | Sort-Object Name | ForEach-Object Name)
if (($actualNames -join "`n") -ne ($expectedNames -join "`n")) {
    $difference = Compare-Object -ReferenceObject $expectedNames -DifferenceObject $actualNames
    throw "Runtime file set is incomplete or contains extras: $($difference | Out-String)"
}

$apertureStart = [Convert]::ToInt64(([string]$runtime.aperture_start).Substring(2), 16)
$apertureEnd = [Convert]::ToInt64(([string]$runtime.aperture_end_exclusive).Substring(2), 16)
$alignment = [long]$runtime.alignment_bytes
$previousEnd = $null
$checkedBytes = 0L
for ($index = 0; $index -lt $segments.Count; $index++) {
    $segment = $segments[$index]
    $expectedName = $expectedNames[$index]
    if ([int]$segment.index -ne $index -or [string]$segment.file -ne $expectedName) {
        throw "Runtime segment index/name mismatch at $index"
    }
    $address = [long]$segment.address
    $nbytes = [long]$segment.nbytes
    $end = $address + $nbytes
    if (($address % $alignment) -ne 0 -or ($nbytes % $alignment) -ne 0) {
        throw "$expectedName is not $alignment-byte aligned"
    }
    if ($address -lt $apertureStart -or $end -gt $apertureEnd) {
        throw "$expectedName escapes the PL-DDR aperture"
    }
    if ($null -ne $previousEnd -and $address -lt [long]$previousEnd) {
        throw "$expectedName overlaps the preceding segment"
    }
    $previousEnd = $end
    $path = Join-Path $modelRoot $expectedName
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -ne $nbytes) {
        throw "$expectedName size mismatch: $size != $nbytes"
    }
    $digest = Get-Sha256 -Path $path
    if ($digest -ne [string]$segment.sha256) {
        throw "$expectedName SHA-256 mismatch: $digest"
    }
    $checkedBytes += $size
}
if ($checkedBytes -ne $expectedBytes) {
    throw "Runtime byte total mismatch: $checkedBytes != $expectedBytes"
}
Write-Host "PASS Q4 runtime $expectedCount segments / $checkedBytes bytes"
Write-Host "PASS portable release audit"

if ($AuditOnly) {
    return
}
if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    throw "Physical launch requires -EvidenceDirectory created by scripts/capture_qweb_uart.py"
}
if (-not (Test-Path -LiteralPath $EvidenceDirectory -PathType Container)) {
    throw "Evidence directory is missing: $EvidenceDirectory"
}
$evidenceRoot = (Resolve-Path -LiteralPath $EvidenceDirectory).Path
$uartRaw = Join-Path $evidenceRoot "uart_raw.bin"
if (-not (Test-Path -LiteralPath $uartRaw -PathType Leaf)) {
    throw "UART capture is not active; missing $uartRaw"
}
$captureClaimPath = Join-Path $evidenceRoot "capture.claim.json"
$captureHeartbeatPath = Join-Path $evidenceRoot "capture.heartbeat.json"
$launchClaimPath = Join-Path $evidenceRoot "launch.claim.json"
$launchJson = Join-Path $evidenceRoot "launch.json"
$startupJson = Join-Path $evidenceRoot "startup.json"
$xsdbLog = Join-Path $evidenceRoot "xsdb.log"
$xsdbProfile = Join-Path $evidenceRoot "xsdb_profile"
if (-not (Test-Path -LiteralPath $captureClaimPath -PathType Leaf)) {
    throw "UART capture claim is missing: $captureClaimPath"
}
if ((Test-Path -LiteralPath $launchClaimPath) -or
    (Test-Path -LiteralPath "$launchClaimPath.tmp") -or
    (Test-Path -LiteralPath $launchJson) -or
    (Test-Path -LiteralPath "$launchJson.tmp") -or
    (Test-Path -LiteralPath $startupJson) -or
    (Test-Path -LiteralPath "$startupJson.tmp") -or
    (Test-Path -LiteralPath $xsdbLog) -or
    (Test-Path -LiteralPath "$xsdbLog.tmp") -or
    (Test-Path -LiteralPath $xsdbProfile)) {
    throw "Refusing to overwrite existing launch evidence in $evidenceRoot"
}

$captureClaim = Get-Content -LiteralPath $captureClaimPath -Raw | ConvertFrom-Json
$captureIdValue = [Guid]::Empty
$capturePid = 0
if ([int]$captureClaim.schema_version -ne 1 -or
    [string]$captureClaim.tool -ne "capture_qweb_uart.py" -or
    -not [Guid]::TryParse([string]$captureClaim.capture_id, [ref]$captureIdValue) -or
    -not [int]::TryParse([string]$captureClaim.pid, [ref]$capturePid) -or
    $capturePid -le 0 -or
    -not [string]::Equals(
        (Resolve-Path -LiteralPath ([string]$captureClaim.evidence_directory)).Path,
        $evidenceRoot,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "UART capture claim is malformed or names another evidence directory"
}
$captureStarted = [DateTimeOffset]::Parse([string]$captureClaim.started_utc)
if ($captureStarted -gt [DateTimeOffset]::UtcNow.AddMinutes(1)) {
    throw "UART capture claim has a future timestamp"
}
$captureClaimSha256 = Get-Sha256 -Path $captureClaimPath
$captureHeartbeat = Assert-CaptureActive `
    -HeartbeatPath $captureHeartbeatPath `
    -EvidenceRoot $evidenceRoot `
    -CaptureId $captureIdValue `
    -CapturePid $capturePid `
    -CaptureStarted $captureStarted

$xsdbPath = Resolve-Xsdb -Requested $Xsdb -DefaultPath ([string]$release.toolchain.default_xsdb)
$vitisRoot = Split-Path -Parent (Split-Path -Parent $xsdbPath)
$auditedToolchain = @()
foreach ($toolFile in $release.toolchain.files) {
    $toolPath = Join-Path $vitisRoot ([string]$toolFile.path)
    if (-not (Test-Path -LiteralPath $toolPath -PathType Leaf)) {
        throw "Required $($release.toolchain.product) file is missing: $toolPath"
    }
    $toolHash = Get-Sha256 -Path $toolPath
    if ($toolHash -ne [string]$toolFile.sha256) {
        throw "Toolchain file SHA-256 mismatch: $toolPath ($toolHash)"
    }
    $auditedToolchain += [ordered]@{
        path = [string]$toolFile.path
        sha256 = $toolHash
    }
}
$expectedXsdb = (Resolve-Path -LiteralPath (Join-Path $vitisRoot "bin\xsdb.bat")).Path
if (-not [string]::Equals($xsdbPath, $expectedXsdb, [StringComparison]::OrdinalIgnoreCase)) {
    throw "-Xsdb must identify the audited Vitis bin/xsdb.bat, not $xsdbPath"
}
if (Test-Path -LiteralPath (Join-Path $vitisRoot "data\baseline.txt")) {
    throw "Refusing Vitis baseline redirection during physical launch"
}
$patchRoot = Join-Path $vitisRoot "patches"
if (Test-Path -LiteralPath $patchRoot -PathType Container) {
    $patchedXsdb = @(Get-ChildItem -LiteralPath $patchRoot -Filter "xsdb.exe" `
        -File -Recurse -ErrorAction Stop)
    if ($patchedXsdb.Count -ne 0) {
        throw "Refusing patched XSDB executable during physical launch: $($patchedXsdb[0].FullName)"
    }
}
$launcherPath = $resolvedArtifacts.launcher
$unsafeEnvironmentNames = @(
    "PATH", "PYTHONPATH", "HOME", "HOMEDRIVE", "HOMEPATH", "HOMESHARE",
    "USERPROFILE", "MYXILINX", "MYVIVADO", "XILINX", "XILINX_PATH",
    "XIL_NO_OVERRIDE", "XIL_PA_NO_DEFAULT_OVERRIDE",
    "XIL_PA_NO_XILINX_OVERRIDE", "XIL_PA_NO_XILINX_SDK_OVERRIDE",
    "XIL_PA_NO_XILINX_PATH_OVERRIDE", "XILINX_VITIS", "XILINX_VIVADO",
    "XILINX_SDK", "XILINX_HLS", "XILINX_VCXX", "XILINX_COMMON_TOOLS",
    "XIL_TPS_ROOT", "RDI_PATCHROOT", "_RDI_BASELINE", "_RDI_NEEDS_PYTHON",
    "RDI_BASELINE", "RDI_PREPEND_PATH", "RDI_MIXED_EXT", "RDI_DEPENDENCY",
    "RDI_BYPASS_ARGS", "RDI_PROG", "RDI_ARGS", "RDI_ARGS_FUNCTION",
    "RDI_SETUP_ENV_FUNCTION", "RDI_JAVALAUNCH", "RDI_VBSLAUNCH",
    "RDI_EXECCLASS", "RDI_CLASSPATH", "RDI_JAVAARGS", "RDI_JAVAFXROOT",
    "RDI_JAVACEFROOT", "RDI_APPROOT", "RDI_BINROOT", "RDI_INSTALLROOT",
    "RDI_PLATFORM", "RDI_OPT_EXT", "_RDI_SETENV_RUN", "TCLLIBPATH",
    "TCL_LIBRARY", "TK_LIBRARY", "QWEB_NETWORK_BIT", "QWEB_NETWORK_XSA",
    "QWEB_NETWORK_FSBL", "QWEB_NETWORK_WEB_ELF", "QWEB_RUNTIME_LOADER",
    "QWEB_HW_SERVER_URL", "QWEB_DEVICE_FILTER"
)
$savedEnvironment = @{}
foreach ($name in $unsafeEnvironmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$markers = @(
    "PASS programmed audited network bitstream",
    "PASS FSBL initialization",
    "PASS PL DDR4 ready status=0x00000005",
    "PASS all 281 QMAP packet headers",
    "PASS a_qweb downloaded and running after audited runtime load"
)
$runId = [Guid]::NewGuid().ToString("D")
$startedUtc = [DateTimeOffset]::UtcNow.ToString("o")
$captureHeartbeat = Assert-CaptureActive `
    -HeartbeatPath $captureHeartbeatPath `
    -EvidenceRoot $evidenceRoot `
    -CaptureId $captureIdValue `
    -CapturePid $capturePid `
    -CaptureStarted $captureStarted
$launchClaim = [ordered]@{
    schema_version = 1
    tool = "run_demo.ps1"
    state = "claimed"
    run_id = $runId
    capture_id = $captureIdValue.ToString("D")
    started_utc = $startedUtc
    evidence_directory = $evidenceRoot
}
Write-JsonExclusive -Path $launchClaimPath -Payload $launchClaim
$launchClaimSha256 = Get-Sha256 -Path $launchClaimPath
[void][System.IO.Directory]::CreateDirectory($xsdbProfile)
if (@(Get-ChildItem -LiteralPath $xsdbProfile -Force).Count -ne 0) {
    throw "Isolated XSDB profile was not empty at creation: $xsdbProfile"
}

$exitCode = $null
$launchFailure = $null
$output = New-Object System.Collections.Generic.List[string]
$locationPushed = $false
try {
    foreach ($name in $unsafeEnvironmentNames) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
    $env:HOME = $xsdbProfile
    $env:USERPROFILE = $xsdbProfile
    $env:QWEB_NETWORK_BIT = $resolvedArtifacts.network_bit
    $env:QWEB_NETWORK_XSA = $resolvedArtifacts.network_xsa
    $env:QWEB_NETWORK_FSBL = $resolvedArtifacts.fsbl
    $env:QWEB_NETWORK_WEB_ELF = $resolvedArtifacts.qweb_elf
    $env:QWEB_RUNTIME_LOADER = $resolvedArtifacts.runtime_loader
    $env:QWEB_HW_SERVER_URL = $HwServerUrl
    $env:QWEB_DEVICE_FILTER = $DeviceFilter

    Push-Location -LiteralPath $xsdbProfile
    $locationPushed = $true
    & $xsdbPath "-no-ini" $launcherPath 2>&1 | ForEach-Object {
        $line = [string]$_
        [void]$output.Add($line)
        Write-Host $line
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "XSDB failed with exit code $exitCode"
    }
    $text = ($output -join "`n") + "`n"
    $cursor = -1
    foreach ($marker in $markers) {
        $position = $text.IndexOf($marker, $cursor + 1, [StringComparison]::Ordinal)
        if ($position -lt 0) {
            throw "XSDB output is missing ordered marker: $marker"
        }
        $cursor = $position
    }
} catch {
    $launchFailure = $_.Exception.Message
} finally {
    if ($locationPushed) {
        Pop-Location
    }
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
}
$text = ($output -join "`n") + "`n"
Write-TextExclusive -Path $xsdbLog -Text $text

$launchReport = [ordered]@{
    schema_version = 1
    tool = "run_demo.ps1"
    run_id = $runId
    capture_id = $captureIdValue.ToString("D")
    passed = ($null -eq $launchFailure)
    failure = $launchFailure
    package_state = [string]$release.package_state
    started_utc = $startedUtc
    completed_utc = [DateTimeOffset]::UtcNow.ToString("o")
    evidence_directory = $evidenceRoot
    capture_claim_sha256 = $captureClaimSha256
    capture_pid = $capturePid
    capture_heartbeat_utc = [string]$captureHeartbeat.heartbeat_utc
    capture_heartbeat_sequence = [long]$captureHeartbeat.sequence
    launch_claim_sha256 = $launchClaimSha256
    release_root = $releaseRoot
    release_manifest_sha256 = Get-Sha256 -Path $manifestPath
    wrapper_sha256 = Get-Sha256 -Path $PSCommandPath
    xsdb = $xsdbPath
    xsdb_sha256 = Get-Sha256 -Path $xsdbPath
    xsdb_exit_code = $exitCode
    xsdb_log_sha256 = Get-Sha256 -Path $xsdbLog
    isolated_profile = "xsdb_profile"
    sanitized_environment = $unsafeEnvironmentNames
    toolchain_product = [string]$release.toolchain.product
    vitis_root = $vitisRoot
    audited_toolchain = $auditedToolchain
    hw_server_url = $HwServerUrl
    runtime_segments = $expectedCount
    runtime_bytes = $checkedBytes
    artifacts = $release.artifacts
    ordered_markers = $markers
}
Write-JsonExclusive -Path $launchJson -Payload $launchReport
if ($null -ne $launchFailure) {
    throw "$launchFailure; see $xsdbLog and $launchJson"
}
Write-Host "PASS portable QWEB physical launch; evidence: $launchJson"
