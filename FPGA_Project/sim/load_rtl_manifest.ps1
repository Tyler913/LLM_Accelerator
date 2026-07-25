function Get-RtlBuildManifest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$RtlDir
    )

    $rtlRoot = [System.IO.Path]::GetFullPath($RtlDir)
    $sourceListPath = Join-Path $rtlRoot "rtl_sources.list"
    $includeListPath = Join-Path $rtlRoot "include_dirs.list"

    foreach ($requiredList in @($sourceListPath, $includeListPath)) {
        if (-not (Test-Path -LiteralPath $requiredList -PathType Leaf)) {
            throw "Missing RTL manifest file: $requiredList"
        }
    }

    $sourceFiles = [System.Collections.Generic.List[string]]::new()
    $sourceSeen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)

    foreach ($rawLine in Get-Content -LiteralPath $sourceListPath) {
        $relativePath = $rawLine.Trim()
        if (($relativePath.Length -eq 0) -or $relativePath.StartsWith("#")) {
            continue
        }

        $extension = [System.IO.Path]::GetExtension($relativePath).ToLowerInvariant()
        if (($extension -ne ".sv") -and ($extension -ne ".v")) {
            throw "RTL source manifest entry must be .sv or .v: $relativePath"
        }

        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $rtlRoot $relativePath))
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "RTL source manifest entry does not exist: $fullPath"
        }
        if (-not $sourceSeen.Add($fullPath)) {
            throw "Duplicate RTL source manifest entry: $relativePath"
        }
        $sourceFiles.Add($fullPath)
    }

    $includeDirs = [System.Collections.Generic.List[string]]::new()
    $includeSeen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)

    foreach ($rawLine in Get-Content -LiteralPath $includeListPath) {
        $relativePath = $rawLine.Trim()
        if (($relativePath.Length -eq 0) -or $relativePath.StartsWith("#")) {
            continue
        }

        $fullPath = [System.IO.Path]::GetFullPath((Join-Path $rtlRoot $relativePath))
        if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
            throw "RTL include manifest entry does not exist: $fullPath"
        }
        if (-not $includeSeen.Add($fullPath)) {
            throw "Duplicate RTL include manifest entry: $relativePath"
        }
        $includeDirs.Add($fullPath)
    }

    $discoveredSources = @(Get-ChildItem -LiteralPath $rtlRoot -Recurse -File |
        Where-Object { $_.Extension -in @(".sv", ".v") } |
        ForEach-Object { $_.FullName })
    $sourceDelta = @(Compare-Object -ReferenceObject @($sourceFiles) `
        -DifferenceObject $discoveredSources)
    if ($sourceDelta.Count -ne 0) {
        $details = ($sourceDelta | ForEach-Object {
            "{0} {1}" -f $_.SideIndicator, $_.InputObject
        }) -join [Environment]::NewLine
        throw "RTL source manifest does not exactly cover the source tree:`n$details"
    }

    $headers = @(Get-ChildItem -LiteralPath $rtlRoot -Recurse -File -Filter "*.svh")
    foreach ($header in $headers) {
        $covered = $false
        foreach ($includeDir in $includeDirs) {
            $prefix = $includeDir.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
                [System.IO.Path]::DirectorySeparatorChar
            if ($header.FullName.StartsWith($prefix,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                $covered = $true
                break
            }
        }
        if (-not $covered) {
            throw "RTL header is not covered by include_dirs.list: $($header.FullName)"
        }
    }

    [pscustomobject]@{
        SourceFiles = @($sourceFiles)
        IncludeDirs = @($includeDirs)
        HeaderFiles = @($headers.FullName)
        SourceListPath = $sourceListPath
        IncludeListPath = $includeListPath
    }
}
