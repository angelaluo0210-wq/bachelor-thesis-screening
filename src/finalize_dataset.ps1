[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName Microsoft.VisualBasic

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ProcessedDir = Join-Path $Root 'data\processed'
$CleanedDir = Join-Path $ProcessedDir 'cleaned'
$CanonicalPath = Join-Path $CleanedDir 'Leenaars_2020_work_qc_cleaned.csv'
$OldPerReviewPath = Join-Path $ProcessedDir 'Leenaars_2020_work_qc_cleaned.csv'
$CombinedPath = Join-Path $ProcessedDir 'all_reviews_cleaned.csv'
$SplitDir = Join-Path $ProcessedDir 'splits'
$ExpectedCanonicalHash = 'B1C4A339B08057D18FEE4F4F3E79531B3166E001492EFD7B9DD38FF3799048D3'
$ExpectedRawHash = '59F97AE4D2A81AB9FA470F6EE00C747F03D90E1F3ACF7A63451590C91756A927'
$Columns = @(
    'doi', 'pmid', 'openalex_id', 'title', 'abstract', 'title_source',
    'abstract_source', 'label_included', 'review_key', '_source_file',
    '_original_row_index'
)

$TrainReviews = @(
    'Appenzeller-Herzog_2019', 'Bos_2018', 'Brouwer_2019', 'Chou_2003',
    'Chou_2004', 'Donners_2021', 'Jeyaraman_2020', 'Leenaars_2020',
    'Meijboom_2021', 'Menon_2022', 'Nelson_2002', 'Oud_2018', 'Sep_2021',
    'Smid_2020', 'van_de_Schoot_2018', 'van_der_Waal_2022', 'Walker_2018',
    'Wolters_2018'
)
$ValidationReviews = @('Hall_2012', 'Moran_2021', 'van_der_Valk_2021', 'Wassenaar_2017')
$TestReviews = @('Leenaars_2019', 'Muthu_2021', 'Radjenovic_2013', 'van_Dis_2020')
$SplitReviews = [ordered]@{
    train = $TrainReviews
    validation = $ValidationReviews
    test = $TestReviews
}
$AllExpectedReviews = @($TrainReviews + $ValidationReviews + $TestReviews | Sort-Object)

function Assert-Equal($Actual, $Expected, [string]$Message) {
    if ($Actual -ne $Expected) {
        throw "$Message (expected=$Expected actual=$Actual)"
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
}

function Assert-Header([string[]]$Header, [string]$Context) {
    if ($Header.Count -eq 0) { throw "Empty header in $Context" }
    $empty = @($Header | Where-Object { [string]::IsNullOrWhiteSpace($_) })
    if ($empty.Count) { throw "Empty header name in $Context" }
    $duplicates = @($Header | Group-Object | Where-Object { $_.Count -gt 1 })
    if ($duplicates.Count) { throw "Duplicate header name in ${Context}: $($duplicates.Name -join ',')" }
}

function Read-CsvMatrix([string]$Path, [string[]]$ExpectedHeader = @(), [switch]$AllowLegacyHeader) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "CSV not found: $Path" }
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser(
        $Path, [System.Text.Encoding]::UTF8, $true
    )
    try {
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(',')
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false
        if ($parser.EndOfData) { throw "Empty CSV: $Path" }
        [string[]]$header = $parser.ReadFields()
        if ($null -eq $header) { throw "Unreadable header: $Path" }
        if (-not $AllowLegacyHeader) { Assert-Header $header $Path }
        if ($ExpectedHeader.Count -gt 0) {
            Assert-Equal ($header -join "`u{001F}") ($ExpectedHeader -join "`u{001F}") "Schema mismatch in $Path"
        }
        $rows = [System.Collections.Generic.List[object]]::new()
        $rowIndex = 0
        while (-not $parser.EndOfData) {
            try { [string[]]$fields = $parser.ReadFields() }
            catch [Microsoft.VisualBasic.FileIO.MalformedLineException] {
                throw "Malformed CSV in $Path near physical line $($parser.ErrorLineNumber)"
            }
            if ($null -eq $fields) { throw "Unreadable row in $Path at logical row $rowIndex" }
            Assert-Equal $fields.Count $header.Count "Unexpected field count in $Path at logical row $rowIndex"
            $rows.Add($fields)
            $rowIndex++
        }
        return [pscustomobject]@{ Header = $header; Rows = $rows }
    }
    finally { $parser.Close() }
}

function Read-CsvHeaderAndRows([string]$Path, [scriptblock]$OnRow) {
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser(
        $Path, [System.Text.Encoding]::UTF8, $true
    )
    try {
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(',')
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false
        if ($parser.EndOfData) { throw "Empty CSV: $Path" }
        [string[]]$header = $parser.ReadFields()
        Assert-Header $header $Path
        Assert-Equal ($header -join "`u{001F}") ($Columns -join "`u{001F}") "Schema mismatch in $Path"
        $rowIndex = 0
        while (-not $parser.EndOfData) {
            try { [string[]]$fields = $parser.ReadFields() }
            catch [Microsoft.VisualBasic.FileIO.MalformedLineException] {
                throw "Malformed CSV in $Path near physical line $($parser.ErrorLineNumber)"
            }
            Assert-Equal $fields.Count $Columns.Count "Unexpected field count in $Path at logical row $rowIndex"
            & $OnRow $fields $rowIndex
            $rowIndex++
        }
        return $rowIndex
    }
    finally { $parser.Close() }
}

function ConvertTo-CsvField([AllowNull()][string]$Value) {
    if ($null -eq $Value) { $Value = '' }
    if ($Value.Contains('"')) { $Value = $Value.Replace('"', '""') }
    if ($Value.IndexOfAny([char[]]@(',', '"', "`r", "`n")) -ge 0) {
        return '"' + $Value + '"'
    }
    return $Value
}

function ConvertTo-CsvLine([string[]]$Fields) {
    if ($Fields.Count -ne $Columns.Count) { throw "Cannot serialize $($Fields.Count) fields into $($Columns.Count)-column schema" }
    return (($Fields | ForEach-Object { ConvertTo-CsvField $_ }) -join ',')
}

function New-Utf8Writer([string]$Path) {
    return New-Object System.IO.StreamWriter($Path, $false, (New-Object System.Text.UTF8Encoding($true)))
}

function Get-ExpectedReview([string]$FileName) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    return ($stem -replace '_(?:final|work_qc)_cleaned$', '')
}

function Normalize-Identifier([string]$Value) {
    return $Value.Trim().TrimEnd('/').ToLowerInvariant()
}

function Normalize-Title([string]$Value) {
    return ([regex]::Replace($Value.ToLowerInvariant(), '[^\p{L}\p{Nd}]+', ' ')).Trim()
}

function New-ReviewStats([string]$Review, [string]$Split) {
    return [ordered]@{
        review_id = $Review; split = $Split; n_records = 0; n_positive = 0
        n_negative = 0; n_other_labels = 0; n_missing_title = 0
        n_missing_abstract = 0; positive_with_missing_abstract = 0
        negative_with_missing_abstract = 0
    }
}

function Add-RowToStats($Stats, [string[]]$Fields) {
    $Stats.n_records++
    $label = $Fields[7]
    if ($label -eq '1') { $Stats.n_positive++ } elseif ($label -eq '0') { $Stats.n_negative++ } else { $Stats.n_other_labels++ }
    if ([string]::IsNullOrWhiteSpace($Fields[3])) { $Stats.n_missing_title++ }
    if ([string]::IsNullOrWhiteSpace($Fields[4])) {
        $Stats.n_missing_abstract++
        if ($label -eq '1') { $Stats.positive_with_missing_abstract++ }
        if ($label -eq '0') { $Stats.negative_with_missing_abstract++ }
    }
}

function Get-StagedCsvStats([string]$Path) {
    $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser(
        $Path, [System.Text.Encoding]::UTF8, $true
    )
    try {
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(',')
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false
        [string[]]$header = $parser.ReadFields()
        Assert-Header $header $Path
        Assert-Equal ($header -join '|') ($Columns -join '|') "Schema mismatch in staged output $Path"
        $byReview = @{}
        $rows = 0; $positive = 0; $negative = 0
        while (-not $parser.EndOfData) {
            try { [string[]]$fields = $parser.ReadFields() }
            catch [Microsoft.VisualBasic.FileIO.MalformedLineException] {
                throw "Malformed staged CSV $Path near physical line $($parser.ErrorLineNumber)"
            }
            Assert-Equal $fields.Count $Columns.Count "Unexpected field count in staged output $Path at logical row $rows"
            $review = $fields[8]
            if ($review -notin $AllExpectedReviews) { throw "Unknown review '$review' in staged output $Path" }
            Assert-RowStructure $fields $review "$Path logical row $rows"
            if (-not $byReview.ContainsKey($review)) {
                $byReview[$review] = [ordered]@{ rows = 0; positive = 0; negative = 0 }
            }
            $byReview[$review].rows++
            if ($fields[7] -eq '1') { $byReview[$review].positive++; $positive++ }
            elseif ($fields[7] -eq '0') { $byReview[$review].negative++; $negative++ }
            $rows++
        }
        return [pscustomobject]@{
            Rows = $rows; Positive = $positive; Negative = $negative; ByReview = $byReview
        }
    }
    finally { $parser.Close() }
}

function Get-SplitForReview([string]$Review) {
    foreach ($name in $SplitReviews.Keys) {
        if ($Review -in $SplitReviews[$name]) { return $name }
    }
    throw "Review has no split assignment: $Review"
}

function Assert-RowStructure([string[]]$Fields, [string]$ExpectedReview, [string]$Context) {
    Assert-Equal $Fields.Count $Columns.Count "Field count mismatch: $Context"
    if ($Fields[7] -notin @('0', '1')) { throw "Unexpected label '$($Fields[7])': $Context" }
    Assert-Equal $Fields[8] $ExpectedReview "Unexpected review_key: $Context"
    if ($Fields[10] -notmatch '^\d+$') { throw "Non-numeric _original_row_index '$($Fields[10])': $Context" }
    $embeddedMetadata = [regex]::new('\uFFFD\?,(?:PubMed|OpenAlex|Missing),(?:PubMed|OpenAlex|Missing),[01],[^\r\n]+(?:\r?\n|$)')
    foreach ($value in $Fields) {
        if ($embeddedMetadata.IsMatch($value)) { throw "Embedded metadata signature: $Context" }
        if ($value -match '\r?\nhttps://doi\.org/[^,]+,https://pubmed\.ncbi\.nlm\.nih\.gov/') {
            throw "Embedded secondary-record signature: $Context"
        }
    }
}

function Assert-NoStableIdLabelConflicts([System.Collections.Generic.List[object]]$Rows, [string]$Review) {
    $labelsByIdentifier = @{}
    foreach ($fields in $Rows) {
        foreach ($columnIndex in @(0, 1, 2)) {
            $value = Normalize-Identifier $fields[$columnIndex]
            if (-not $value) { continue }
            $key = "$columnIndex|$value"
            if (-not $labelsByIdentifier.ContainsKey($key)) { $labelsByIdentifier[$key] = @{} }
            $labelsByIdentifier[$key][$fields[7]] = $true
        }
    }
    $conflicts = @($labelsByIdentifier.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 })
    if ($conflicts.Count) { throw "Unresolved stable-identifier label conflicts in ${Review}: $($conflicts.Name -join ';')" }
}

function Get-ReviewDigest([System.Collections.Generic.List[object]]$Rows) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        foreach ($fields in $Rows) {
            $bytes = [System.Text.Encoding]::UTF8.GetBytes((ConvertTo-CsvLine $fields) + "`n")
            $null = $sha.TransformBlock($bytes, 0, $bytes.Length, $bytes, 0)
        }
        $null = $sha.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return ([System.BitConverter]::ToString($sha.Hash)).Replace('-', '')
    }
    finally { $sha.Dispose() }
}

# Locate the immutable raw file without depending on the console code page.
$rawRoots = @(Get-ChildItem -LiteralPath $Root -Directory | Where-Object { $_.Name -like '*(raw)' })
Assert-Equal $rawRoots.Count 1 'Expected exactly one raw directory matching *(raw)'
$rawCandidates = @(Get-ChildItem -LiteralPath $rawRoots[0].FullName -Recurse -File -Filter 'Leenaars_2020_work_qc.csv')
Assert-Equal $rawCandidates.Count 1 'Expected exactly one Leenaars_2020 raw CSV'
$RawPath = $rawCandidates[0].FullName

foreach ($path in @($CanonicalPath, $OldPerReviewPath, $CombinedPath, $RawPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required input not found: $path" }
}
Assert-Equal (Get-Sha256 $CanonicalPath) $ExpectedCanonicalHash 'Canonical Leenaars file is not the audited 7,214-row version'
Assert-Equal (Get-Sha256 $RawPath) $ExpectedRawHash 'Raw source changed since forensic audit'

$cleanedFiles = @(Get-ChildItem -LiteralPath $CleanedDir -File -Filter '*.csv' | Sort-Object Name)
Assert-Equal $cleanedFiles.Count 26 'Expected exactly 26 canonical cleaned review files'
$observedReviews = @($cleanedFiles | ForEach-Object { Get-ExpectedReview $_.Name } | Sort-Object)
Assert-Equal ($observedReviews -join '|') ($AllExpectedReviews -join '|') 'Canonical review list does not match fixed 26-review split'

$protectedHashes = @{}
foreach ($file in $cleanedFiles) {
    if ($file.FullName -ne $CanonicalPath) { $protectedHashes[$file.FullName] = Get-Sha256 $file.FullName }
}
$protectedHashes[$RawPath] = Get-Sha256 $RawPath
$auditFiles = @(Get-ChildItem -LiteralPath (Join-Path $Root 'reports') -File -Filter '*.csv')
foreach ($file in $auditFiles) { $protectedHashes[$file.FullName] = Get-Sha256 $file.FullName }

$tempDir = Join-Path $ProcessedDir ('.finalize_tmp_' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $tempDir
$stagedCanonical = Join-Path $tempDir 'Leenaars_2020_work_qc_cleaned.csv'
$stagedCombined = Join-Path $tempDir 'all_reviews_cleaned.csv'
$stagedTrain = Join-Path $tempDir 'train.csv'
$stagedValidation = Join-Path $tempDir 'validation.csv'
$stagedTest = Join-Path $tempDir 'test.csv'
$stagedSummary = Join-Path $tempDir 'split_summary.csv'

try {
    # Step 1: restore raw logical row 292, cross-checked against the older local
    # per-review processed copy, and stage the 7,215-row canonical file.
    $raw = Read-CsvMatrix $RawPath -AllowLegacyHeader
    Assert-Equal $raw.Header.Count 25 'Unexpected raw Leenaars schema width'
    Assert-Equal $raw.Rows.Count 7207 'Unexpected raw Leenaars row count'
    [string[]]$raw292 = $raw.Rows[292]
    Assert-Equal $raw292[0] '' 'Unexpected row 292 DOI'
    Assert-Equal $raw292[1] 'https://pubmed.ncbi.nlm.nih.gov/3532399' 'Unexpected row 292 PMID'
    Assert-Equal $raw292[2] 'https://openalex.org/W2416186387' 'Unexpected row 292 OpenAlex ID'
    Assert-Equal $raw292[7] '1' 'Unexpected row 292 label'
    Assert-Equal $raw292[8] 'Leenaars_2020' 'Unexpected row 292 review'

    $old = Read-CsvMatrix $OldPerReviewPath -AllowLegacyHeader
    Assert-Equal $old.Header.Count 25 'Unexpected old per-review schema width'
    $oldMatches = @($old.Rows | Where-Object { $_[1] -eq $raw292[1] -and $_[2] -eq $raw292[2] })
    Assert-Equal $oldMatches.Count 1 'Row 292 stable identifiers must match exactly one old per-review record'
    for ($i = 0; $i -lt 9; $i++) { Assert-Equal $oldMatches[0][$i] $raw292[$i] "Row 292 upstream mismatch at field $i" }

    $canonical = Read-CsvMatrix $CanonicalPath $Columns
    Assert-Equal $canonical.Rows.Count 7214 'Unexpected Leenaars row count before row 292 restore'
    Assert-Equal @($canonical.Rows | Where-Object { $_[10] -eq '292' }).Count 0 'Row 292 already exists in canonical'
    $restored292 = [string[]]@(
        $raw292[0], $raw292[1], $raw292[2], $raw292[3], $raw292[4],
        $raw292[5], $raw292[6], $raw292[7], $raw292[8],
        'Leenaars_2020_work_qc.csv', '292'
    )
    Assert-RowStructure $restored292 'Leenaars_2020' 'restored row 292'

    $leenaarsRows = [System.Collections.Generic.List[object]]::new()
    $inserted = $false
    foreach ($fields in $canonical.Rows) {
        if (-not $inserted -and [int]$fields[10] -gt 292) {
            $leenaarsRows.Add($restored292)
            $inserted = $true
        }
        $leenaarsRows.Add($fields)
    }
    if (-not $inserted) { $leenaarsRows.Add($restored292) }
    Assert-Equal $leenaarsRows.Count 7215 'Leenaars row count after row 292 restore must be 7,215'
    Assert-Equal @($leenaarsRows | Where-Object { $_[10] -eq '292' }).Count 1 'Restored row 292 cardinality mismatch'
    foreach ($fields in $leenaarsRows) { Assert-RowStructure $fields 'Leenaars_2020' "Leenaars original row $($fields[10])" }
    Assert-NoStableIdLabelConflicts $leenaarsRows 'Leenaars_2020'

    foreach ($idIndex in @(0, 1, 2)) {
        $needle = Normalize-Identifier $restored292[$idIndex]
        if (-not $needle) { continue }
        $matches = @($leenaarsRows | Where-Object { (Normalize-Identifier $_[$idIndex]) -eq $needle })
        Assert-Equal $matches.Count 1 "Unexpected stable-identifier duplicate introduced by row 292 (column $idIndex)"
    }
    $titleMatches = @($leenaarsRows | Where-Object { (Normalize-Title $_[3]) -eq (Normalize-Title $restored292[3]) })
    Assert-Equal $titleMatches.Count 2 'Expected one known normalized-title collision for restored row 292'
    $titleOther = @($titleMatches | Where-Object { $_[10] -ne '292' })
    Assert-Equal $titleOther.Count 1 'Unexpected normalized-title collision cardinality for row 292'
    Assert-Equal $titleOther[0][10] '5159' 'Normalized-title collision is not the audited row 5159 record'
    if ($titleOther[0][1] -eq $restored292[1] -or $titleOther[0][2] -eq $restored292[2]) {
        throw 'Audited title collision unexpectedly shares a stable identifier with row 292'
    }

    $writer = New-Utf8Writer $stagedCanonical
    try {
        $writer.WriteLine(($Columns | ForEach-Object { ConvertTo-CsvField $_ }) -join ',')
        foreach ($fields in $leenaarsRows) { $writer.WriteLine((ConvertTo-CsvLine $fields)) }
    }
    finally { $writer.Dispose() }
    $canonicalCheck = Read-CsvMatrix $stagedCanonical $Columns
    Assert-Equal $canonicalCheck.Rows.Count 7215 'Staged Leenaars canonical row count mismatch'

    # Validate each canonical source and write the exact same serialized row to
    # combined plus its one fixed split. All outputs remain staged until QC passes.
    $statsByReview = @{}
    $totalRows = 0; $totalPositive = 0; $totalNegative = 0
    $combinedWriter = New-Utf8Writer $stagedCombined
    $splitWriters = @{
        train = New-Utf8Writer $stagedTrain
        validation = New-Utf8Writer $stagedValidation
        test = New-Utf8Writer $stagedTest
    }
    try {
        $headerLine = ($Columns | ForEach-Object { ConvertTo-CsvField $_ }) -join ','
        $combinedWriter.WriteLine($headerLine)
        foreach ($name in $splitWriters.Keys) { $splitWriters[$name].WriteLine($headerLine) }
        foreach ($file in $cleanedFiles) {
            $review = Get-ExpectedReview $file.Name
            $split = Get-SplitForReview $review
            $sourcePath = if ($review -eq 'Leenaars_2020') { $stagedCanonical } else { $file.FullName }
            $matrix = Read-CsvMatrix $sourcePath $Columns
            $stats = New-ReviewStats $review $split
            foreach ($fields in $matrix.Rows) {
                Assert-RowStructure $fields $review "$($file.Name) original row $($fields[10])"
                Add-RowToStats $stats $fields
                $line = ConvertTo-CsvLine $fields
                $combinedWriter.WriteLine($line)
                $splitWriters[$split].WriteLine($line)
            }
            Assert-NoStableIdLabelConflicts $matrix.Rows $review
            Assert-Equal $stats.n_other_labels 0 "Other label detected in $review"
            $statsByReview[$review] = $stats
            $totalRows += $stats.n_records
            $totalPositive += $stats.n_positive
            $totalNegative += $stats.n_negative
        }
    }
    finally {
        $combinedWriter.Dispose()
        foreach ($name in $splitWriters.Keys) { $splitWriters[$name].Dispose() }
    }
    Assert-Equal ($totalPositive + $totalNegative) $totalRows 'Combined label accounting mismatch before concat'

    $summaryColumns = @(
        'review_id','split','n_records','n_positive','n_negative','n_other_labels',
        'positive_rate','has_missing_title','n_missing_title','has_missing_abstract',
        'n_missing_abstract','positive_with_missing_abstract','negative_with_missing_abstract'
    )
    $summaryWriter = New-Utf8Writer $stagedSummary
    try {
        $summaryWriter.WriteLine(($summaryColumns | ForEach-Object { ConvertTo-CsvField $_ }) -join ',')
        foreach ($split in @('train','validation','test')) {
            foreach ($review in @($SplitReviews[$split] | Sort-Object)) {
                $stats = $statsByReview[$review]
                $rate = if ($stats.n_records) { ([double]$stats.n_positive / [double]$stats.n_records).ToString('G17', [Globalization.CultureInfo]::InvariantCulture) } else { '0' }
                [string[]]$summaryFields = @(
                    $review, $split, [string]$stats.n_records, [string]$stats.n_positive,
                    [string]$stats.n_negative, [string]$stats.n_other_labels, $rate,
                    $(if ($stats.n_missing_title) { 'True' } else { 'False' }),
                    [string]$stats.n_missing_title,
                    $(if ($stats.n_missing_abstract) { 'True' } else { 'False' }),
                    [string]$stats.n_missing_abstract,
                    [string]$stats.positive_with_missing_abstract,
                    [string]$stats.negative_with_missing_abstract
                )
                $summaryWriter.WriteLine((($summaryFields | ForEach-Object { ConvertTo-CsvField $_ }) -join ','))
            }
        }
    }
    finally { $summaryWriter.Dispose() }

    # Independent streaming reread verifies schema, cardinality, labels,
    # per-review counts, complete assignment, and absence of overlap.
    $combinedCheck = Get-StagedCsvStats $stagedCombined
    Assert-Equal $combinedCheck.Rows $totalRows 'Staged combined row count differs from 26-file sum'
    $combinedReviews = @($combinedCheck.ByReview.Keys | Sort-Object)
    Assert-Equal ($combinedReviews -join '|') ($AllExpectedReviews -join '|') 'Staged combined review coverage mismatch'
    $combinedPositive = $combinedCheck.Positive
    $combinedNegative = $combinedCheck.Negative
    Assert-Equal $combinedPositive $totalPositive 'Staged combined positive count mismatch'
    Assert-Equal $combinedNegative $totalNegative 'Staged combined negative count mismatch'
    Assert-Equal ($combinedPositive + $combinedNegative) $combinedCheck.Rows 'Staged combined contains other labels'
    foreach ($review in $AllExpectedReviews) {
        Assert-Equal $combinedCheck.ByReview[$review].rows $statsByReview[$review].n_records "Combined row count mismatch for $review"
        Assert-Equal $combinedCheck.ByReview[$review].positive $statsByReview[$review].n_positive "Combined positive count mismatch for $review"
        Assert-Equal $combinedCheck.ByReview[$review].negative $statsByReview[$review].n_negative "Combined negative count mismatch for $review"
    }

    $splitPathByName = @{ train=$stagedTrain; validation=$stagedValidation; test=$stagedTest }
    $splitStats = @{}
    $seenSplitReviews = @{}
    $splitRowsTotal = 0; $splitPositiveTotal = 0; $splitNegativeTotal = 0
    foreach ($split in @('train','validation','test')) {
        $matrix = Get-StagedCsvStats $splitPathByName[$split]
        $reviews = @($matrix.ByReview.Keys | Sort-Object)
        Assert-Equal ($reviews -join '|') ((@($SplitReviews[$split] | Sort-Object)) -join '|') "$split review assignment mismatch"
        foreach ($review in $reviews) {
            if ($seenSplitReviews.ContainsKey($review)) { throw "Review overlap across splits: $review" }
            $seenSplitReviews[$review] = $split
            Assert-Equal $matrix.ByReview[$review].rows $statsByReview[$review].n_records "Split row count mismatch for $review"
            Assert-Equal $matrix.ByReview[$review].positive $statsByReview[$review].n_positive "Split positive count mismatch for $review"
            Assert-Equal $matrix.ByReview[$review].negative $statsByReview[$review].n_negative "Split negative count mismatch for $review"
        }
        $positive = $matrix.Positive
        $negative = $matrix.Negative
        Assert-Equal ($positive + $negative) $matrix.Rows "$split contains other labels"
        $splitStats[$split] = [ordered]@{
            reviews = $reviews.Count; rows = $matrix.Rows; positive = $positive
            negative = $negative; positive_rate = if ($matrix.Rows) { $positive / $matrix.Rows } else { 0 }
        }
        $splitRowsTotal += $matrix.Rows
        $splitPositiveTotal += $positive
        $splitNegativeTotal += $negative
    }
    Assert-Equal $seenSplitReviews.Count 26 'Not all 26 reviews assigned exactly once'
    Assert-Equal $splitRowsTotal $totalRows 'Split row preservation failure'
    Assert-Equal $splitPositiveTotal $totalPositive 'Split positive preservation failure'
    Assert-Equal $splitNegativeTotal $totalNegative 'Split negative preservation failure'
    $summaryCheck = Read-CsvMatrix $stagedSummary $summaryColumns
    Assert-Equal $summaryCheck.Rows.Count 26 'Split summary must contain exactly 26 review rows'

    # All QC passed: replace only the user-authorized outputs.
    if (-not (Test-Path -LiteralPath $SplitDir -PathType Container)) { $null = New-Item -ItemType Directory -Path $SplitDir }
    Move-Item -LiteralPath $stagedCanonical -Destination $CanonicalPath -Force
    Move-Item -LiteralPath $stagedCombined -Destination $CombinedPath -Force
    Move-Item -LiteralPath $stagedTrain -Destination (Join-Path $SplitDir 'train.csv') -Force
    Move-Item -LiteralPath $stagedValidation -Destination (Join-Path $SplitDir 'validation.csv') -Force
    Move-Item -LiteralPath $stagedTest -Destination (Join-Path $SplitDir 'test.csv') -Force
    Move-Item -LiteralPath $stagedSummary -Destination (Join-Path $SplitDir 'split_summary.csv') -Force

    foreach ($path in $protectedHashes.Keys) {
        Assert-Equal (Get-Sha256 $path) $protectedHashes[$path] "Protected input changed unexpectedly: $path"
    }

    [ordered]@{
        leenaars_rows_before = 7214
        leenaars_rows_after = 7215
        row292_restored = $true
        combined_rows = $totalRows
        combined_positive = $totalPositive
        combined_negative = $totalNegative
        combined_reviews = 26
        train = $splitStats.train
        validation = $splitStats.validation
        test = $splitStats.test
        combined_sha256 = Get-Sha256 $CombinedPath
        train_sha256 = Get-Sha256 (Join-Path $SplitDir 'train.csv')
        validation_sha256 = Get-Sha256 (Join-Path $SplitDir 'validation.csv')
        test_sha256 = Get-Sha256 (Join-Path $SplitDir 'test.csv')
    } | ConvertTo-Json -Depth 4
}
finally {
    if (Test-Path -LiteralPath $tempDir -PathType Container) {
        $resolvedTemp = [System.IO.Path]::GetFullPath($tempDir)
        $resolvedProcessed = [System.IO.Path]::GetFullPath($ProcessedDir).TrimEnd('\') + '\'
        if (-not $resolvedTemp.StartsWith($resolvedProcessed, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean temp directory outside processed root: $resolvedTemp"
        }
        $tempItem = Get-Item -LiteralPath $resolvedTemp -Force
        if ($tempItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to clean reparse-point temp directory: $resolvedTemp"
        }
        $children = @(Get-ChildItem -LiteralPath $resolvedTemp -Force)
        $unexpectedDirectories = @($children | Where-Object { $_.PSIsContainer })
        if ($unexpectedDirectories.Count) { throw "Unexpected subdirectory in temp cleanup: $($unexpectedDirectories.FullName -join ',')" }
        foreach ($child in $children) { Remove-Item -LiteralPath $child.FullName -Force }
        Remove-Item -LiteralPath $resolvedTemp -Force
    }
}
