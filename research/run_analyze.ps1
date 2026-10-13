# Opens in its own window: offline analysis, then two live checks (the deployed
# operating point, and the mid-frontier operating point where the probe cascade runs),
# with progress bars and log copies.
param([string]$Run = "main", [string]$Models = "")
$host.UI.RawUI.WindowTitle = "Router research: analysis and live checks for run '$Run'"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$log = "research\runs\$Run"
$extra = @()
if ($Models) { $extra = @("--models") + $Models.Split(",") }

Write-Host "Step 1/5: python -m research.analyze --run $Run $extra" -ForegroundColor Cyan
& $py -m research.analyze --run $Run @extra 2> "$log\analyze_stderr.log" | Tee-Object -FilePath "$log\analyze.log"
"analyze exit $LASTEXITCODE" | Out-File "$log\analyze_done.txt"

Write-Host "Step 2/5: live check at the deployed operating point" -ForegroundColor Cyan
& $py -m research.live_check --run $Run 2> "$log\live_check_stderr.log" | Tee-Object -FilePath "$log\live_check.log"

$mid = & $py -c "import json; print(json.load(open('results/router_eval.json'))['irt']['mid_operating_point']['lambda'])"
Write-Host "Step 3/5: live check at the mid-frontier operating point (lambda $mid)" -ForegroundColor Cyan
$env:ROUTER_LAMBDA = $mid
& $py -m research.live_check --run $Run --out router_live_check_mid.json 2> "$log\live_check_mid_stderr.log" | Tee-Object -FilePath "$log\live_check_mid.log"
Write-Host "Step 4/5: live check with every query starting on the cascade's small model (lambda 50)" -ForegroundColor Cyan
$env:ROUTER_LAMBDA = "50"
& $py -m research.live_check --run $Run --out router_live_check_cascade.json 2> "$log\live_check_cascade_stderr.log" | Tee-Object -FilePath "$log\live_check_cascade.log"
Remove-Item Env:ROUTER_LAMBDA
Write-Host "Step 5/5: Ollama model residency and reload check" -ForegroundColor Cyan
& $py -m research.eviction_check 2> "$log\eviction_stderr.log" | Tee-Object -FilePath "$log\eviction_check.log"
"live checks done" | Out-File "$log\live_check_done.txt"
Write-Host "Done." -ForegroundColor Green
