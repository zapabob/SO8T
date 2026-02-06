Get-ChildItem -Path 'H:\from_D\webdataset' -Filter '*.jsonl' -Recurse -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -gt 1000 } |
  Sort-Object Length -Descending |
  ForEach-Object { "$([math]::Round($_.Length/1024,1)) KB`t$($_.FullName)" }
