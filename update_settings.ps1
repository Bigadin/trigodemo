$path = Join-Path $env:APPDATA 'Cursor\User\settings.json'
$content = Get-Content $path -Raw
$json = $content | ConvertFrom-Json
$json | Add-Member -NotePropertyName 'workbench.sideBar.location' -NotePropertyValue 'left' -Force
$json | ConvertTo-Json -Depth 10 | Set-Content $path -Encoding UTF8
Write-Host "Done"
