# Sync assets et code entre worktree vpi et dossier Documents
# Usage: .\scripts\sync-assets.ps1 [direction]
#   direction: "to-documents" (default) | "to-vpi" | "both"
# - to-documents: copie index.js, index.html, CLAUDE.md de vpi vers Documents
# - to-vpi: copie les SVG de Documents vers vpi
# - both: fait les deux (recommandé après modifs)

param([string]$Direction = "both")

$Docs = "c:\Users\ysdol\Documents\GitHub\trigodemo"
$Vpi = "c:\Users\ysdol\.cursor\worktrees\trigodemo\vpi"
$SvgDir = "static\assets_youn\SvIcons\SVGnew"
$SvgFiles = @('Yhumidity.svg','Yobstruction.svg','Yfissure.svg','Ycountingppl.svg','Yheatmapdense.svg','Ytraj.svg','Ypublic transport.svg','Ytransport2.svg','Yhumancat.svg')

if ($Direction -eq "to-documents" -or $Direction -eq "both") {
    Write-Host "Sync vpi -> Documents (index.js, index.html, CLAUDE.md)"
    Copy-Item "$Vpi\static\js\index.js" "$Docs\static\js\index.js" -Force
    Copy-Item "$Vpi\static\index.html" "$Docs\static\index.html" -Force
    Copy-Item "$Vpi\CLAUDE.md" "$Docs\CLAUDE.md" -Force
}

if ($Direction -eq "to-vpi" -or $Direction -eq "both") {
    Write-Host "Sync Documents -> vpi (SVG icons)"
    foreach ($s in $SvgFiles) {
        if (Test-Path "$Docs\$SvgDir\$s") {
            Copy-Item "$Docs\$SvgDir\$s" "$Vpi\$SvgDir\$s" -Force
            Write-Host "  $s"
        }
    }
}

Write-Host "Done."
