$ErrorActionPreference='Stop'
$dest=Join-Path $PSScriptRoot 'illustrative_cases.md'
if (Test-Path -LiteralPath $dest) { throw 'Refusing to overwrite illustrative cases' }
$rows=Import-Csv -LiteralPath (Join-Path $PSScriptRoot 'qwen_fn_audit.csv') -Encoding UTF8 | Where-Object { $_.assessment -ne 'NOT_ASSESSED' }
$lines=[Collections.Generic.List[string]]::new()
$lines.Add('# Illustrative cases: partial exploratory assessment')
$lines.Add('')
$lines.Add('These 11 selected examples are not representative and are not independently adjudicated label errors. A is an apparent explicit conflict, B is ambiguous. The article and criteria quotations are checked exact substrings of the extracted fields. The complete abstract and criteria are in the CSV.')
foreach($r in $rows) { $lines.Add(''); $lines.Add("## $($r.review_key) / source row $($r.source_row_index) / $($r.assessment)"); $lines.Add(''); $lines.Add("- ID: $($r.record_id)"); $lines.Add("- Title: $($r.title)"); $lines.Add("- Article evidence: `"$($r.article_evidence)`""); $lines.Add("- Criteria evidence: `"$($r.criteria_evidence)`""); $lines.Add("- Assessment: $($r.explanation)") }
[IO.File]::WriteAllLines($dest,$lines,[Text.UTF8Encoding]::new($false))
