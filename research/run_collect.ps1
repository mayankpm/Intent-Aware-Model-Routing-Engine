# Opens in its own window: runs the collector with a live progress bar and a log copy.
param([string]$Run = "main", [string]$Temperature = "0", [string]$Seed = "0")
$host.UI.RawUI.WindowTitle = "Router research: collecting run '$Run'"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force "research\runs\$Run" | Out-Null
$env:PYTHONUNBUFFERED = "1"
& .\.venv\Scripts\python.exe -m research.collect --run $Run --temperature $Temperature --seed $Seed 2> "research\runs\$Run\collect_stderr.log" | Tee-Object -FilePath "research\runs\$Run\collect.log" -Append
"exit $LASTEXITCODE" | Out-File "research\runs\$Run\collect_done.txt"
