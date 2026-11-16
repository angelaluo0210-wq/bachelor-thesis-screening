$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$predPath = Join-Path $root 'supplementary_202610/llm_baseline/qwen_final_test_20261029/predictions.jsonl'
$testPath = Join-Path $root 'data/processed/splits/test.csv'
$matches = @(Get-ChildItem -LiteralPath (Join-Path $root 'data') -File | Where-Object { $_.Name -like 'eligibility_criteria*.xlsx' })
if ($matches.Count -ne 1) { throw 'Expected exactly one criteria workbook' }
$criteriaPath = $matches[0].FullName
$outputs = @('qwen_fn_audit.csv','summary_by_review.csv','summary_by_category.csv','audit_report.md','illustrative_cases.md')
foreach ($name in $outputs) { if (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name)) { throw "Refusing to overwrite $name" } }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($criteriaPath)
function Read-Entry($name) { $reader = [IO.StreamReader]::new($zip.GetEntry($name).Open()); try { $reader.ReadToEnd() } finally { $reader.Dispose() } }
try { [xml]$stringsXml = Read-Entry 'xl/sharedStrings.xml'; [xml]$sheet = Read-Entry 'xl/worksheets/sheet1.xml' } finally { $zip.Dispose() }
$ns = [Xml.XmlNamespaceManager]::new($sheet.NameTable); $ns.AddNamespace('x','http://schemas.openxmlformats.org/spreadsheetml/2006/main')
$sns = [Xml.XmlNamespaceManager]::new($stringsXml.NameTable); $sns.AddNamespace('x','http://schemas.openxmlformats.org/spreadsheetml/2006/main')
$strings = @($stringsXml.SelectNodes('//x:si',$sns) | ForEach-Object { (@($_.SelectNodes('.//x:t',$sns) | ForEach-Object { $_.InnerText }) -join '') })
$criteria = @{}
foreach ($row in $sheet.SelectNodes('//x:sheetData/x:row',$ns)) {
  $cells = @{}; foreach ($c in $row.SelectNodes('x:c',$ns)) { $col = $c.r -replace '\d',''; $v=$c.SelectSingleNode('x:v',$ns); if ($null -ne $v) { if ($c.t -eq 's') { $cells[$col]=$strings[[int]$v.InnerText] } else { $cells[$col]=$v.InnerText } } }
  if ($cells.A -and $cells.A -ne 'Review key') { $criteria[$cells.A.ToLowerInvariant()]=$cells }
}
$test = @(Import-Csv -LiteralPath $testPath -Encoding UTF8)
$pred = @(Get-Content -LiteralPath $predPath -Encoding UTF8 | ForEach-Object { if ($_.Trim()) { $_ | ConvertFrom-Json } })
if ($test.Count -ne 23592 -or $pred.Count -ne 23592) { throw 'Population count differs from manifest' }
$seen = @{}; $usedIndices = @{}; $fn = [Collections.Generic.List[object]]::new(); $bad=0
foreach ($p in $pred) {
  if ($seen.ContainsKey($p.record_id)) { throw "Duplicate ID: $($p.record_id)" }; $seen[$p.record_id]=1
  $i=[int]$p.source_row_index; if ($i -lt 0 -or $i -ge $test.Count -or $usedIndices.ContainsKey($i)) { throw "Invalid or duplicate source index $i" }; $usedIndices[$i]=1
  $t=$test[$i]; $parts=[Collections.Generic.List[string]]::new(); $parts.Add(($t.review_key.Trim().ToLowerInvariant() -replace '[- ]','_'))
  foreach($field in @('pmid','doi','openalex_id')) { $value=([string]$t.$field).Trim().ToLowerInvariant(); if ($value) { $parts.Add("$field=$value") } }
  if ($parts.Count -eq 1) { $parts.Add("source_row_index=$i") }
  if (($parts -join '|') -ne $p.record_id -or $t.review_key -ne $p.review_name -or [int]$t.label_included -ne [int]$p.true_label) { $bad++; continue }
  if ([int]$p.true_label -eq 1 -and $p.status -eq 'completed' -and $p.decision -eq 'exclude') {
    $c=$criteria[$t.review_key.ToLowerInvariant()]; if (!$c) { throw "No criteria row for $($t.review_key)" }
    $inc=[string]$c.F; $exc=[string]$c.G; if (-not $inc.Trim() -and -not $exc.Trim()) { $inc=[string]$c.E }
    $fn.Add([pscustomobject][ordered]@{review_key=$t.review_key;record_id=$p.record_id;source_row_index=$i;title=$t.title;abstract=$t.abstract;abstract_missing=[string]::IsNullOrWhiteSpace($t.abstract);gold_label=$t.label_included;qwen_status=$p.status;qwen_decision=$p.decision;inclusion_criteria=$inc;exclusion_criteria=$exc;inclusion_missing=[string]::IsNullOrWhiteSpace($inc);exclusion_missing=[string]::IsNullOrWhiteSpace($exc);assessment='NOT_ASSESSED';explanation='Not yet individually assessed; no inference from missing information.';article_evidence='';criteria_evidence=''})
  }
}
if ($bad -ne 0) { throw "Alignment mismatches: $bad" }
$fn | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'qwen_fn_audit.csv') -NoTypeInformation -Encoding UTF8
$fn | Group-Object review_key | ForEach-Object { [pscustomobject]@{review_key=$_.Name;fn_count=$_.Count;percent_of_all_fn=[math]::Round(100*$_.Count/$fn.Count,2);missing_abstract=(@($_.Group | Where-Object abstract_missing -eq $true)).Count;assessment_status='not assessed'} } | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'summary_by_review.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{assessment='NOT_ASSESSED';count=$fn.Count;percent=100;note='Not an A/B/C/D category; individual adjudication has not been performed.'} | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'summary_by_category.csv') -NoTypeInformation -Encoding UTF8
Write-Output "predictions=$($pred.Count) unique=$($seen.Count) aligned=$($usedIndices.Count) mismatches=$bad FN=$($fn.Count)"
$fn | Group-Object review_key | ForEach-Object { Write-Output "$($_.Name): $($_.Count); missing abstracts: $(@($_.Group | Where-Object abstract_missing -eq $true).Count)" }
foreach ($key in @('Muthu_2021','Leenaars_2019','Radjenovic_2013','van_Dis_2020')) { $c=$criteria[$key.ToLowerInvariant()]; Write-Output "CRITERIA $key`nINCLUSION: $($c.F)`nEXCLUSION: $($c.G)" }
