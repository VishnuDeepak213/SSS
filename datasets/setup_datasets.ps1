# Dataset Download Manager for SDS
# Creates folder structure and provides download instructions

Write-Host ("=" * 70)
Write-Host "SDS Dataset Download Manager - 50GB Plan"
Write-Host ("=" * 70)
Write-Host ""

# Create directory structure
Write-Host "[1/3] Creating dataset directories..."
$datasets = @(
    "datasets/coco_person",
    "datasets/coco_person/train2017",
    "datasets/coco_person/val2017",
    "datasets/coco_person/annotations",
    "datasets/mot17",
    "datasets/mot17/train",
    "datasets/mot17/test",
    "datasets/crowdhuman",
    "datasets/crowdhuman/train",
    "datasets/crowdhuman/val",
    "datasets/shanghaitech",
    "datasets/shanghaitech/part_A",
    "datasets/shanghaitech/part_B",
    "datasets/umn",
    "datasets/ucy_eth",
    "datasets/ucy_eth/ucy",
    "datasets/ucy_eth/eth",
    "datasets/custom",
    "datasets/custom/normal",
    "datasets/custom/dense",
    "datasets/custom/events",
    "datasets/processed",
    "datasets/downloads"
)

foreach ($dir in $datasets) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

# Check available disk space
Write-Host "[2/3] Checking disk space..."
$drive = (Get-Location).Drive.Name
$disk = Get-PSDrive $drive
$freeSpaceGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "Available space on ${drive}: $freeSpaceGB GB" -ForegroundColor Cyan

if ($freeSpaceGB -lt 60) {
    Write-Host "⚠ Warning: Less than 60GB free. Recommended: 60GB+" -ForegroundColor Yellow
} else {
    Write-Host "✓ Sufficient space available" -ForegroundColor Green
}
Write-Host ""

# Download instructions
Write-Host "[3/3] Dataset Download Instructions"
Write-Host ("=" * 70)
Write-Host ""

Write-Host "AUTOMATIC DOWNLOADS (Python script available):" -ForegroundColor Green
Write-Host "  • UMN Dataset (~200MB)" -ForegroundColor White
Write-Host "  • UCY/ETH trajectories (~800MB)" -ForegroundColor White
Write-Host ""
Write-Host "Run: python datasets/download_auto.py" -ForegroundColor Cyan
Write-Host ""

Write-Host "MANUAL DOWNLOADS REQUIRED:" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. COCO Person Subset (~5GB) - ESSENTIAL" -ForegroundColor White
Write-Host "   Download train2017 images:" -ForegroundColor Gray
Write-Host "   http://images.cocodataset.org/zips/train2017.zip" -ForegroundColor Blue
Write-Host "   Download val2017 images:" -ForegroundColor Gray
Write-Host "   http://images.cocodataset.org/zips/val2017.zip" -ForegroundColor Blue
Write-Host "   Download annotations:" -ForegroundColor Gray
Write-Host "   http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -ForegroundColor Blue
Write-Host "   Extract to: datasets/coco_person/" -ForegroundColor Gray
Write-Host "   Then filter person class: python scripts/filter_coco_person.py" -ForegroundColor Cyan
Write-Host ""

Write-Host "2. MOT17 (~5GB) - ESSENTIAL [REGISTRATION REQUIRED]" -ForegroundColor White
Write-Host "   Step 1: Register at https://motchallenge.net/" -ForegroundColor Gray
Write-Host "   Step 2: Download MOT17 from:" -ForegroundColor Gray
Write-Host "   https://motchallenge.net/data/MOT17/" -ForegroundColor Blue
Write-Host "   Extract to: datasets/mot17/" -ForegroundColor Gray
Write-Host ""

Write-Host "3. CrowdHuman (~15GB) - ESSENTIAL" -ForegroundColor White
Write-Host "   Download all train + val files from:" -ForegroundColor Gray
Write-Host "   https://www.crowdhuman.org/download.html" -ForegroundColor Blue
Write-Host "   Files needed:" -ForegroundColor Gray
Write-Host "     - CrowdHuman_train01.zip (4.6GB)" -ForegroundColor Gray
Write-Host "     - CrowdHuman_train02.zip (4.7GB)" -ForegroundColor Gray
Write-Host "     - CrowdHuman_train03.zip (4.6GB)" -ForegroundColor Gray
Write-Host "     - CrowdHuman_val.zip (1.7GB)" -ForegroundColor Gray
Write-Host "     - annotation_train.odgt" -ForegroundColor Gray
Write-Host "     - annotation_val.odgt" -ForegroundColor Gray
Write-Host "   Extract to: datasets/crowdhuman/" -ForegroundColor Gray
Write-Host ""

Write-Host "4. ShanghaiTech (~350MB) - ESSENTIAL" -ForegroundColor White
Write-Host "   Download from Google Drive:" -ForegroundColor Gray
Write-Host "   https://drive.google.com/file/d/16dhJn7k4FWVFTvvHEUbwDDldxKqR6V_4/view" -ForegroundColor Blue
Write-Host "   Or GitHub: https://github.com/desenzhou/ShanghaiTechDataset" -ForegroundColor Blue
Write-Host "   Extract to: datasets/shanghaitech/" -ForegroundColor Gray
Write-Host ""

Write-Host ("=" * 70)
Write-Host ""
Write-Host "DOWNLOAD CHECKLIST:" -ForegroundColor Cyan
Write-Host "  [ ] Run automatic downloader (Python script)" -ForegroundColor White
Write-Host "  [ ] Download COCO (5GB)" -ForegroundColor White
Write-Host "  [ ] Register & download MOT17 (5GB)" -ForegroundColor White
Write-Host "  [ ] Download CrowdHuman (15GB)" -ForegroundColor White
Write-Host "  [ ] Download ShanghaiTech (0.35GB)" -ForegroundColor White
Write-Host ""
Write-Host "Total download size: ~26GB (+ 5-10GB after processing)" -ForegroundColor Yellow
Write-Host "Estimated download time: 2-6 hours (depends on connection)" -ForegroundColor Yellow
Write-Host ""
Write-Host "TIP: Download large files (COCO, CrowdHuman) overnight!" -ForegroundColor Green
Write-Host ("=" * 70)
