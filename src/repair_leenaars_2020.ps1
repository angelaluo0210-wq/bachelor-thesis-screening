[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName Microsoft.VisualBasic

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$rawRoots = @(Get-ChildItem -LiteralPath $Root -Directory | Where-Object { $_.Name -like '*(raw)' })
if ($rawRoots.Count -ne 1) { throw "Expected one project raw directory matching *(raw), found $($rawRoots.Count)" }
$rawCandidates = @(Get-ChildItem -LiteralPath $rawRoots[0].FullName -Recurse -File -Filter 'Leenaars_2020_work_qc.csv')
if ($rawCandidates.Count -ne 1) { throw "Expected one Leenaars_2020 raw CSV, found $($rawCandidates.Count)" }
$RawPath = $rawCandidates[0].FullName
$CanonicalPath = Join-Path $Root 'data\processed\cleaned\Leenaars_2020_work_qc_cleaned.csv'
$CombinedPath = Join-Path $Root 'data\processed\all_reviews_cleaned.csv'
$OldPerReviewPath = Join-Path $Root 'data\processed\Leenaars_2020_work_qc_cleaned.csv'
$ForensicAuditPath = Join-Path $Root 'reports\leenaars_2020_misalignment_audit.csv'
$RepairReportPath = Join-Path $Root 'reports\leenaars_2020_repair_report.csv'
$Row292ReportPath = Join-Path $Root 'reports\leenaars_2020_row292_audit.csv'

$ExpectedPreRepairHash = 'A25D28E0347EF50C4D090C883120A2443F2E5781874045ADFAA2DBBD479E6289'
$ExpectedRawHash = '59F97AE4D2A81AB9FA470F6EE00C747F03D90E1F3ACF7A63451590C91756A927'
$ExpectedCombinedHash = '555969A3663A3957EDB234F692FD01A75915CFD85DAE176F1F30A6A5092E345B'
$TargetIndexes = @(3528, 4406, 5692, 5902, 5985, 5990, 6217, 6362, 6493, 6661, 6960, 7115)
$TwoArticleIndexes = @(4406, 5692, 5985, 5990, 6217, 6362, 6493, 6960)
$NoAbstractPrimary = @(5902, 5985, 6217, 6362)
$NoAbstractSecondary = @(5692)
$OutputColumns = @(
    'doi', 'pmid', 'openalex_id', 'title', 'abstract', 'title_source',
    'abstract_source', 'label_included', 'review_key', '_source_file',
    '_original_row_index'
)

$SecondaryArticles = @{
    4406 = @{ Doi = 'https://doi.org/10.1161/circulationaha.107.734384'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/18427133'; OpenAlex = 'https://openalex.org/W2126340219'; Title = 'Angiotensin-converting enzyme inhibition improves vascular function in rheumatoid arthritis.' }
    5692 = @{ Doi = 'https://doi.org/10.1093/rheumatology/38.2.97'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/10342619'; OpenAlex = 'https://openalex.org/W2052377647'; Title = 'Corticosteroids in rheumatoid arthritis--the case against.' }
    5985 = @{ Doi = 'https://doi.org/10.1002/art.24836'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/19877093'; OpenAlex = 'https://openalex.org/W2101139377'; Title = 'Does rheumatoid arthritis equal diabetes mellitus as an independent risk factor for cardiovascular disease? A prospective study.' }
    5990 = @{ Doi = 'https://doi.org/10.1016/s1297-319x(10)70001-4'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/20610315'; OpenAlex = 'https://openalex.org/W1968115891'; Title = 'Tocilizumab: therapy and safety management.' }
    6217 = @{ Doi = 'https://doi.org/10.1517/14656566.2013.825606'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/24044485'; OpenAlex = 'https://openalex.org/W2004568544'; Title = 'New and emerging treatments for osteoarthritis management: will the dream come true with personalized medicine?' }
    6362 = @{ Doi = 'https://doi.org/10.2174/157016309789869010'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/20025592'; OpenAlex = 'https://openalex.org/W2075960462'; Title = 'Biologics as treatment for systemic lupus: great efforts, sobering results, new challenges.' }
    6493 = @{ Doi = 'https://doi.org/10.1016/s0140-6736(07)60784-3'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/17570481'; OpenAlex = 'https://openalex.org/W2062697678'; Title = 'New therapies for treatment of rheumatoid arthritis.' }
    6960 = @{ Doi = 'https://doi.org/10.1517/14728214.13.1.175'; Pmid = 'https://pubmed.ncbi.nlm.nih.gov/18321156'; OpenAlex = 'https://openalex.org/W2023461539'; Title = 'Emerging drugs for rheumatoid arthritis.' }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
}

function Assert-Equal($Actual, $Expected, [string]$Message) {
    if ($Actual -ne $Expected) {
        throw "$Message (expected=$Expected actual=$Actual)"
    }
}

function Read-CsvMatrix([string]$Path, [Nullable[int]]$ExpectedFieldCount = $null) {
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser(
        $Path,
        [System.Text.Encoding]::UTF8,
        $true
    )
    try {
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(',')
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false
        if ($parser.EndOfData) { throw "CSV is empty: $Path" }
        [string[]]$header = $parser.ReadFields()
        if ($null -eq $header) { throw "CSV header is unreadable: $Path" }
        if ($null -ne $ExpectedFieldCount) {
            Assert-Equal $header.Count ([int]$ExpectedFieldCount) "Unexpected header field count in $Path"
        }
        $rows = [System.Collections.Generic.List[object]]::new()
        $logicalRow = 0
        while (-not $parser.EndOfData) {
            try {
                [string[]]$fields = $parser.ReadFields()
            }
            catch [Microsoft.VisualBasic.FileIO.MalformedLineException] {
                throw "Malformed CSV record in $Path near physical line $($parser.ErrorLineNumber): $($parser.ErrorLine)"
            }
            if ($null -eq $fields) { throw "Unreadable CSV record in $Path at logical row $logicalRow" }
            $expected = if ($null -ne $ExpectedFieldCount) { [int]$ExpectedFieldCount } else { $header.Count }
            Assert-Equal $fields.Count $expected "Unexpected field count in $Path at logical row $logicalRow"
            $rows.Add($fields)
            $logicalRow++
        }
        return [pscustomobject]@{ Header = $header; Rows = $rows }
    }
    finally {
        $parser.Close()
    }
}

function Assert-UniqueNonEmptyHeader([string[]]$Header, [string]$Context) {
    if ($Header.Count -eq 0) { throw "Empty header in $Context" }
    $seen = @{}
    foreach ($column in $Header) {
        if ([string]::IsNullOrWhiteSpace($column)) { throw "Empty header name in $Context" }
        if ($seen.ContainsKey($column)) { throw "Duplicate header '$column' in $Context" }
        $seen[$column] = $true
    }
}

function Convert-MatrixToRecords($Matrix, [string]$Context) {
    Assert-UniqueNonEmptyHeader $Matrix.Header $Context
    $records = [System.Collections.Generic.List[object]]::new()
    foreach ($fields in $Matrix.Rows) {
        $record = [ordered]@{}
        for ($i = 0; $i -lt $Matrix.Header.Count; $i++) {
            $record[$Matrix.Header[$i]] = $fields[$i]
        }
        $records.Add($record)
    }
    return $records
}

function ConvertTo-CsvField([AllowNull()][string]$Value) {
    if ($null -eq $Value) { $Value = '' }
    if ($Value.Contains('"')) { $Value = $Value.Replace('"', '""') }
    if ($Value.IndexOfAny([char[]]@(',', '"', "`r", "`n")) -ge 0) {
        return '"' + $Value + '"'
    }
    return $Value
}

function Write-CsvStrict([string]$Path, [string[]]$Columns, $Rows) {
    Assert-UniqueNonEmptyHeader $Columns $Path
    $encoding = New-Object System.Text.UTF8Encoding($true)
    $writer = New-Object System.IO.StreamWriter($Path, $false, $encoding)
    try {
        $writer.WriteLine((($Columns | ForEach-Object { ConvertTo-CsvField $_ }) -join ','))
        $rowNumber = 0
        foreach ($row in $Rows) {
            $keys = @($row.Keys)
            if ($keys.Count -ne $Columns.Count) {
                throw "Unexpected field count while writing $Path row $rowNumber"
            }
            foreach ($key in $keys) {
                if ($key -notin $Columns) { throw "Unexpected field '$key' while writing $Path row $rowNumber" }
            }
            foreach ($column in $Columns) {
                if (-not $row.Contains($column)) { throw "Missing field '$column' while writing $Path row $rowNumber" }
            }
            $writer.WriteLine((($Columns | ForEach-Object { ConvertTo-CsvField ([string]$row[$_]) }) -join ','))
            $rowNumber++
        }
    }
    finally {
        $writer.Dispose()
    }
}

function New-OutputRow(
    [string]$Doi, [string]$Pmid, [string]$OpenAlex, [string]$Title,
    [string]$Abstract, [string]$TitleSource, [string]$AbstractSource,
    [string]$Label, [string]$Review, [string]$SourceFile, [string]$OriginalIndex
) {
    return [ordered]@{
        doi = $Doi; pmid = $Pmid; openalex_id = $OpenAlex; title = $Title
        abstract = $Abstract; title_source = $TitleSource; abstract_source = $AbstractSource
        label_included = $Label; review_key = $Review; _source_file = $SourceFile
        _original_row_index = $OriginalIndex
    }
}

function Copy-NormalCanonicalRow([string[]]$Fields) {
    for ($i = 9; $i -le 24; $i++) {
        if (-not [string]::IsNullOrEmpty($Fields[$i])) {
            throw "Unexpected data in unnamed canonical column $i at original row $($Fields[26])"
        }
    }
    return New-OutputRow $Fields[0] $Fields[1] $Fields[2] $Fields[3] $Fields[4] $Fields[5] $Fields[6] $Fields[7] $Fields[8] $Fields[25] $Fields[26]
}

function New-SingleRepair([int]$Index, [string[]]$RawFields) {
    if ($Index -eq 5902) {
        $abstract = ''
        $titleSource = $RawFields[4]
        $abstractSource = $RawFields[5]
    }
    else {
        $match = [regex]::Match(
            $RawFields[4],
            '^(?<abstract>.*)\uFFFD\?(?<titleSource>PubMed|OpenAlex|Missing)$',
            [System.Text.RegularExpressions.RegexOptions]::Singleline
        )
        if (-not $match.Success) { throw "Single-record suffix not found at raw row $Index" }
        $abstract = $match.Groups['abstract'].Value
        $titleSource = $match.Groups['titleSource'].Value
        $abstractSource = $RawFields[5]
    }
    Assert-Equal $RawFields[6] '0' "Recovered label mismatch at raw row $Index"
    Assert-Equal $RawFields[7] 'Leenaars_2020' "Recovered review mismatch at raw row $Index"
    $pmid = $RawFields[1]
    if ($Index -eq 5902) { $pmid = 'https://pubmed.ncbi.nlm.nih.gov/16261703' }
    return New-OutputRow $RawFields[0] $pmid $RawFields[2] $RawFields[3] $abstract $titleSource $abstractSource '0' 'Leenaars_2020' 'Leenaars_2020_work_qc.csv' ([string]$Index)
}

function New-TwoArticleRepair([int]$Index, [string[]]$RawFields) {
    $newline = $RawFields[4].IndexOf("`n")
    if ($newline -lt 0) { throw "Embedded second record not found at raw row $Index" }
    $primaryPayload = $RawFields[4].Substring(0, $newline).TrimEnd("`r")
    $primaryMatch = [regex]::Match(
        $primaryPayload,
        '^(?<abstract>.*)\uFFFD\?,(?<titleSource>PubMed|OpenAlex|Missing),(?<abstractSource>PubMed|OpenAlex|Missing),(?<label>0),(?<review>Leenaars_2020)$',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if (-not $primaryMatch.Success) { throw "Primary embedded metadata marker not found at raw row $Index" }
    $primaryAbstract = $primaryMatch.Groups['abstract'].Value
    $primaryAbstractSource = $primaryMatch.Groups['abstractSource'].Value
    if ($Index -in $NoAbstractPrimary) {
        $primaryAbstract = ''
        $primaryAbstractSource = 'Missing'
    }
    $primaryPmid = $RawFields[1]
    if ($Index -eq 6217) { $primaryPmid = 'https://pubmed.ncbi.nlm.nih.gov/15790644' }
    $primary = New-OutputRow $RawFields[0] $primaryPmid $RawFields[2] $RawFields[3] $primaryAbstract $primaryMatch.Groups['titleSource'].Value $primaryAbstractSource '0' 'Leenaars_2020' 'Leenaars_2020_work_qc.csv' ([string]$Index)

    $secondaryInfo = $SecondaryArticles[$Index]
    if ($null -eq $secondaryInfo) { throw "Missing confirmed secondary metadata for raw row $Index" }
    $payloadParts = [System.Collections.Generic.List[string]]::new()
    $payloadParts.Add($RawFields[4].Substring($newline + 1))
    for ($i = 5; $i -lt $RawFields.Count; $i++) { $payloadParts.Add($RawFields[$i]) }
    $secondaryPayload = ($payloadParts -join ',').TrimEnd(',')
    $identifierPrefix = "$($secondaryInfo.Doi),$($secondaryInfo.Pmid),$($secondaryInfo.OpenAlex),"
    if (-not $secondaryPayload.StartsWith($identifierPrefix, [System.StringComparison]::Ordinal)) {
        throw "Secondary identifier prefix mismatch at raw row $Index"
    }
    $remainder = $secondaryPayload.Substring($identifierPrefix.Length)
    if (-not $remainder.StartsWith($secondaryInfo.Title, [System.StringComparison]::Ordinal)) {
        throw "Secondary title mismatch at raw row $Index"
    }
    $remainder = $remainder.Substring($secondaryInfo.Title.Length)
    if ($remainder.StartsWith('"')) { $remainder = $remainder.Substring(1) }
    if (-not $remainder.StartsWith(',')) { throw "Secondary title delimiter mismatch at raw row $Index" }
    $remainder = $remainder.Substring(1)
    $secondaryMatch = [regex]::Match(
        $remainder,
        '^(?<abstract>.*),(?<titleSource>PubMed|OpenAlex|Missing),(?<abstractSource>PubMed|OpenAlex|Missing),(?<label>0),(?<review>Leenaars_2020)$',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if (-not $secondaryMatch.Success) { throw "Secondary metadata tail mismatch at raw row $Index" }
    $secondaryAbstract = $secondaryMatch.Groups['abstract'].Value.TrimEnd('"')
    $secondaryAbstractSource = $secondaryMatch.Groups['abstractSource'].Value
    if ($Index -in $NoAbstractSecondary) {
        $secondaryAbstract = ''
        $secondaryAbstractSource = 'Missing'
    }
    $secondary = New-OutputRow $secondaryInfo.Doi $secondaryInfo.Pmid $secondaryInfo.OpenAlex $secondaryInfo.Title $secondaryAbstract $secondaryMatch.Groups['titleSource'].Value $secondaryAbstractSource '0' 'Leenaars_2020' 'Leenaars_2020_work_qc.csv' ([string]$Index)
    return @($primary, $secondary)
}

function Normalize-Identifier([string]$Value) {
    return $Value.Trim().TrimEnd('/').ToLowerInvariant()
}

function Normalize-Title([string]$Value) {
    return ([regex]::Replace($Value.ToLowerInvariant(), '[^\p{L}\p{Nd}]+', ' ')).Trim()
}

function Get-DuplicateEvidence($Candidate, $AllRows, [int]$CandidatePosition) {
    $matches = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $AllRows.Count; $i++) {
        if ($i -eq $CandidatePosition) { continue }
        $other = $AllRows[$i]
        $signals = [System.Collections.Generic.List[string]]::new()
        foreach ($field in @('doi', 'pmid', 'openalex_id')) {
            $left = Normalize-Identifier ([string]$Candidate[$field])
            $right = Normalize-Identifier ([string]$other[$field])
            if ($left -and $left -eq $right) { $signals.Add("$field=$left") }
        }
        $leftTitle = Normalize-Title ([string]$Candidate['title'])
        $rightTitle = Normalize-Title ([string]$other['title'])
        if ($leftTitle -and $leftTitle -eq $rightTitle) { $signals.Add("normalized_title=$leftTitle") }
        if ($signals.Count -gt 0) {
            $matches.Add("original_row_index=$($other['_original_row_index']):$($signals -join '|')")
        }
    }
    return ($matches -join '; ')
}

# Fail closed unless this is exactly the audited pre-repair state.
foreach ($required in @($RawPath, $CanonicalPath, $CombinedPath, $OldPerReviewPath, $ForensicAuditPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file not found: $required" }
}
Assert-Equal (Get-Sha256 $CanonicalPath) $ExpectedPreRepairHash 'Canonical file is not the audited pre-repair version'
Assert-Equal (Get-Sha256 $RawPath) $ExpectedRawHash 'Raw source hash changed since forensic QC'
Assert-Equal (Get-Sha256 $CombinedPath) $ExpectedCombinedHash 'Combined dataset hash changed since forensic QC'

$protectedBefore = @{}
Get-ChildItem -LiteralPath (Join-Path $Root 'data') -Recurse -File -Filter '*.csv' | ForEach-Object {
    if ([System.IO.Path]::GetFullPath($_.FullName) -ne [System.IO.Path]::GetFullPath($CanonicalPath)) {
        $protectedBefore[$_.FullName] = Get-Sha256 $_.FullName
    }
}
$protectedBefore[$RawPath] = Get-Sha256 $RawPath

$raw = Read-CsvMatrix $RawPath 25
$canonical = Read-CsvMatrix $CanonicalPath 27
Assert-Equal $raw.Rows.Count 7207 'Unexpected raw row count'
Assert-Equal $canonical.Rows.Count 7206 'Unexpected canonical pre-repair row count'

$forensic = Convert-MatrixToRecords (Read-CsvMatrix $ForensicAuditPath) $ForensicAuditPath
Assert-Equal $forensic.Count 12 'Forensic audit must contain exactly 12 target rows'
$auditTargets = @($forensic | ForEach-Object { [int]$_['original_row_index'] } | Sort-Object)
Assert-Equal ($auditTargets -join ',') (($TargetIndexes | Sort-Object) -join ',') 'Forensic audit target set mismatch'
foreach ($record in $forensic) {
    Assert-Equal $record['correct_review_key'] 'Leenaars_2020' 'Forensic review conclusion mismatch'
    Assert-Equal $record['correct_label'] '0' 'Forensic label conclusion mismatch'
    Assert-Equal $record['confidence'] 'high' 'Forensic confidence mismatch'
}

$canonicalByIndex = @{}
foreach ($fields in $canonical.Rows) {
    $index = [int]$fields[26]
    if ($canonicalByIndex.ContainsKey($index)) { throw "Duplicate pre-repair original index: $index" }
    $canonicalByIndex[$index] = $fields
}
if ($canonicalByIndex.ContainsKey(292)) { throw 'Row 292 unexpectedly exists in canonical pre-repair data' }
foreach ($index in $TargetIndexes) {
    if (-not $canonicalByIndex.ContainsKey($index)) { throw "Target original index missing from canonical: $index" }
}

$outputRowsWithMetadata = [System.Collections.Generic.List[object]]::new()
$recoveredPositions = [System.Collections.Generic.List[object]]::new()
foreach ($fields in $canonical.Rows) {
    $index = [int]$fields[26]
    if ($index -notin $TargetIndexes) {
        $outputRowsWithMetadata.Add([pscustomobject]@{ Row = (Copy-NormalCanonicalRow $fields); RepairPosition = 0 })
        continue
    }
    [string[]]$rawFields = $raw.Rows[$index]
    if ($index -in $TwoArticleIndexes) {
        $repaired = @(New-TwoArticleRepair $index $rawFields)
    }
    else {
        $repaired = @((New-SingleRepair $index $rawFields))
    }
    for ($position = 0; $position -lt $repaired.Count; $position++) {
        $outputPosition = $outputRowsWithMetadata.Count
        $outputRowsWithMetadata.Add([pscustomobject]@{ Row = $repaired[$position]; RepairPosition = $position + 1 })
        $recoveredPositions.Add([pscustomobject]@{ OutputPosition = $outputPosition; SourceIndex = $index; ArticlePosition = $position + 1 })
    }
}

$outputRows = [System.Collections.Generic.List[object]]::new()
foreach ($entry in $outputRowsWithMetadata) { $outputRows.Add($entry.Row) }
Assert-Equal $outputRows.Count 7214 'Post-repair row count mismatch'
Assert-Equal ($outputRows.Count - $canonical.Rows.Count) 8 'Net row change mismatch'
Assert-Equal $recoveredPositions.Count 20 'Recovered article count mismatch'

$embeddedSignature = [regex]::new('\uFFFD\?,(?:PubMed|OpenAlex|Missing),(?:PubMed|OpenAlex|Missing),[01],Leenaars_2020(?:\r?\n|$)')
foreach ($row in $outputRows) {
    if ([string]$row['label_included'] -notin @('0', '1')) { throw "Unexpected label at original row $($row['_original_row_index'])" }
    if ([string]$row['review_key'] -ne 'Leenaars_2020') { throw "Unexpected review key at original row $($row['_original_row_index'])" }
    foreach ($column in $OutputColumns) {
        $value = [string]$row[$column]
        if ($embeddedSignature.IsMatch($value)) { throw "Embedded metadata signature remains at original row $($row['_original_row_index']) field $column" }
        if ($value -match '\r?\nhttps://doi\.org/[^,]+,https://pubmed\.ncbi\.nlm\.nih\.gov/') {
            throw "Embedded second-record signature remains at original row $($row['_original_row_index']) field $column"
        }
    }
}

$repairReportRows = [System.Collections.Generic.List[object]]::new()
$possibleDuplicateCount = 0
foreach ($recovered in $recoveredPositions) {
    $row = $outputRows[$recovered.OutputPosition]
    if ([string]::IsNullOrWhiteSpace([string]$row['title'])) { throw "Recovered article has an empty title at original row $($row['_original_row_index'])" }
    $duplicateEvidence = Get-DuplicateEvidence $row $outputRows $recovered.OutputPosition
    $possibleDuplicate = if ($duplicateEvidence) { 'YES' } else { 'NO' }
    if ($duplicateEvidence) { $possibleDuplicateCount++ }
    $abstractStatus = if ([string]::IsNullOrEmpty([string]$row['abstract'])) { 'missing_confirmed_by_source' } else { 'recovered_from_upstream_payload' }
    $action = if ($recovered.SourceIndex -in $TwoArticleIndexes) { 'split_merged_record' } elseif ($recovered.SourceIndex -eq 5902) { 'restore_missing_abstract_shift' } else { 'restore_single_record_shift' }
    $notes = if ($recovered.SourceIndex -in @(6661, 7115)) { 'Pre-existing Japanese mojibake preserved; text repair remains separate.' } elseif ([string]::IsNullOrEmpty([string]$row['abstract'])) { 'No formal abstract; no abstract text was generated.' } else { '' }
    $reportRow = [ordered]@{
        source_original_row_index = [string]$recovered.SourceIndex
        article_position = [string]$recovered.ArticlePosition
        title = [string]$row['title']
        doi = [string]$row['doi']
        pmid = [string]$row['pmid']
        openalex_id = [string]$row['openalex_id']
        review_key = [string]$row['review_key']
        label_included = [string]$row['label_included']
        abstract_status = $abstractStatus
        repair_action = $action
        evidence = 'Confirmed local upstream payload and reports/leenaars_2020_misalignment_audit.csv'
        confidence = 'high'
        notes = $notes
        possible_duplicate = $possibleDuplicate
        duplicate_evidence = $duplicateEvidence
    }
    $repairReportRows.Add($reportRow)
}

$row292 = $raw.Rows[292]
Assert-Equal $row292[0] '' 'Unexpected row 292 DOI'
Assert-Equal $row292[1] 'https://pubmed.ncbi.nlm.nih.gov/3532399' 'Unexpected row 292 PMID'
Assert-Equal $row292[2] 'https://openalex.org/W2416186387' 'Unexpected row 292 OpenAlex ID'
Assert-Equal $row292[7] '1' 'Unexpected row 292 label'
Assert-Equal $row292[8] 'Leenaars_2020' 'Unexpected row 292 review key'
$row292ReportRows = @([ordered]@{
    original_row_index = '292'
    doi = $row292[0]
    pmid = $row292[1]
    openalex_id = $row292[2]
    title = $row292[3]
    abstract = $row292[4]
    title_source = $row292[5]
    abstract_source = $row292[6]
    label_included = $row292[7]
    review_key = $row292[8]
    found_in_raw = 'YES'
    found_in_old_per_review_processed = 'YES'
    found_in_old_combined = 'YES'
    found_in_canonical_cleaned = 'NO'
    other_versions_found = 'raw work_qc; data/processed/Leenaars_2020_work_qc_cleaned.csv; data/processed/all_reviews_cleaned.csv; conflict and duplicate audit reports'
    likely_reason_category = 'F. build/version inconsistency'
    evidence = 'Structurally valid raw record and stable identifiers; current builder documents no deduplication/conflict deletion; normalized-title-only conflict with original row 5159 has different PMID and OpenAlex ID; no deletion audit exists.'
    recommended_action = 'restore'
    confidence = 'high'
    exact_disappearance_mechanism_confidence = 'medium'
    notes = 'Do not restore in this repair. Restore only in a separately authorized version-reconciliation change, then rebuild downstream artifacts.'
})

$repairColumns = @('source_original_row_index','article_position','title','doi','pmid','openalex_id','review_key','label_included','abstract_status','repair_action','evidence','confidence','notes','possible_duplicate','duplicate_evidence')
$row292Columns = @('original_row_index','doi','pmid','openalex_id','title','abstract','title_source','abstract_source','label_included','review_key','found_in_raw','found_in_old_per_review_processed','found_in_old_combined','found_in_canonical_cleaned','other_versions_found','likely_reason_category','evidence','recommended_action','confidence','exact_disappearance_mechanism_confidence','notes')

$canonicalTemp = Join-Path (Split-Path $CanonicalPath) ('.' + [System.IO.Path]::GetFileName($CanonicalPath) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
$repairTemp = Join-Path (Split-Path $RepairReportPath) ('.' + [System.IO.Path]::GetFileName($RepairReportPath) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
$row292Temp = Join-Path (Split-Path $Row292ReportPath) ('.' + [System.IO.Path]::GetFileName($Row292ReportPath) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
try {
    Write-CsvStrict $canonicalTemp $OutputColumns $outputRows
    Write-CsvStrict $repairTemp $repairColumns $repairReportRows
    Write-CsvStrict $row292Temp $row292Columns $row292ReportRows

    $stagedCanonical = Read-CsvMatrix $canonicalTemp 11
    Assert-UniqueNonEmptyHeader $stagedCanonical.Header $canonicalTemp
    Assert-Equal $stagedCanonical.Rows.Count 7214 'Staged canonical row count mismatch'
    $stagedRepair = Read-CsvMatrix $repairTemp $repairColumns.Count
    Assert-UniqueNonEmptyHeader $stagedRepair.Header $repairTemp
    Assert-Equal $stagedRepair.Rows.Count 20 'Staged repair report row count mismatch'
    $staged292 = Read-CsvMatrix $row292Temp $row292Columns.Count
    Assert-UniqueNonEmptyHeader $staged292.Header $row292Temp
    Assert-Equal $staged292.Rows.Count 1 'Staged row 292 report row count mismatch'

    Move-Item -LiteralPath $canonicalTemp -Destination $CanonicalPath -Force
    Move-Item -LiteralPath $repairTemp -Destination $RepairReportPath -Force
    Move-Item -LiteralPath $row292Temp -Destination $Row292ReportPath -Force
}
finally {
    foreach ($temp in @($canonicalTemp, $repairTemp, $row292Temp)) {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    }
}

$final = Read-CsvMatrix $CanonicalPath 11
Assert-UniqueNonEmptyHeader $final.Header $CanonicalPath
Assert-Equal $final.Rows.Count 7214 'Final canonical row count mismatch'
foreach ($path in $protectedBefore.Keys) {
    Assert-Equal (Get-Sha256 $path) $protectedBefore[$path] "Protected CSV changed unexpectedly: $path"
}

[ordered]@{
    canonical_file = $CanonicalPath
    rows_before = 7206
    rows_after = 7214
    delta = 8
    malformed_source_rows_repaired = 12
    articles_expected = 20
    articles_recovered = 20
    recovered_label_0 = 20
    recovered_label_1 = 0
    possible_duplicates = $possibleDuplicateCount
    final_sha256 = Get-Sha256 $CanonicalPath
    repair_report = $RepairReportPath
    row292_report = $Row292ReportPath
} | ConvertTo-Json
