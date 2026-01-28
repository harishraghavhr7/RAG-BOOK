Param()

# Sync files from Retrieval Augmentation App frontend into fullstack-rag/frontend
$src = Join-Path $PSScriptRoot '..\Retrieval Augmentation App'
$dst = Join-Path $PSScriptRoot '..\fullstack-rag\frontend'

Write-Host "Source: $src"
Write-Host "Destination: $dst"

if (-not (Test-Path $src)) {
    Write-Error "Source folder not found: $src"
    exit 1
}

# Ensure destination exists
New-Item -ItemType Directory -Path $dst -Force | Out-Null

# Copy all files and folders, overwrite
Write-Host "Copying files..."
robocopy $src $dst /MIR /Z /W:1 /R:2 | Out-Null

if ($LASTEXITCODE -ge 8) {
    Write-Error "robocopy failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Sync complete."
Write-Host "Next: cd $dst ; npm install ; npm run dev"
