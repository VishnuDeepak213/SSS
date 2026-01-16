# Dataset Download Quick Reference

## 🚀 Quick Start

### Step 1: Setup folders
```powershell
.\datasets\setup_datasets.ps1
```

### Step 2: Download public datasets automatically
```powershell
python datasets\download_auto.py
```

### Step 3: Download manual datasets (see below)

---

## 📥 Manual Downloads (Essential for Training)

### 1. COCO Person (~5GB) - ESSENTIAL

**Download links** (copy-paste to browser):
```
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

**After download**:
- Extract train2017.zip → `datasets/coco_person/train2017/`
- Extract val2017.zip → `datasets/coco_person/val2017/`
- Extract annotations → `datasets/coco_person/annotations/`

---

### 2. MOT17 (~5GB) - ESSENTIAL

**Registration required**: https://motchallenge.net/

**Steps**:
1. Register account at motchallenge.net
2. Go to: https://motchallenge.net/data/MOT17/
3. Download `MOT17.zip`
4. Extract to `datasets/mot17/`

---

### 3. CrowdHuman (~15GB) - ESSENTIAL

**Direct download**: https://www.crowdhuman.org/download.html

**Files to download**:
- CrowdHuman_train01.zip (4.6GB)
- CrowdHuman_train02.zip (4.7GB)
- CrowdHuman_train03.zip (4.6GB)
- CrowdHuman_val.zip (1.7GB)
- annotation_train.odgt
- annotation_val.odgt

**After download**:
- Extract all ZIPs to `datasets/crowdhuman/`

**Pro tip**: Use download manager (IDM, FDM) for faster parallel downloads!

---

### 4. ShanghaiTech (~350MB) - ESSENTIAL

**Option 1 - Google Drive**:
https://drive.google.com/file/d/16dhJn7k4FWVFTvvHEUbwDDldxKqR6V_4/view

**Option 2 - GitHub**:
https://github.com/desenzhou/ShanghaiTechDataset

**After download**:
- Extract to `datasets/shanghaitech/`
- Should contain `part_A/` and `part_B/` folders

---

## 🔄 Alternative Download Methods

### Using wget (PowerShell)

```powershell
# Download COCO train images
wget http://images.cocodataset.org/zips/train2017.zip -OutFile datasets\downloads\train2017.zip

# Download COCO annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile datasets\downloads\annotations.zip
```

### Using curl (PowerShell)

```powershell
# Download with progress
curl -L "http://images.cocodataset.org/zips/train2017.zip" -o "datasets\downloads\train2017.zip"
```

### Using aria2c (fastest - multi-connection)

```powershell
# Install: scoop install aria2
# Or: choco install aria2

# Download with 16 connections (fastest!)
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2017.zip -d datasets\downloads
```

---

## 📊 Download Checklist

```
Essential (Must have):
[ ] COCO Person (~5GB)
[ ] MOT17 (~5GB)
[ ] CrowdHuman (~15GB)
[ ] ShanghaiTech (~350MB)

Supplementary (Auto-downloaded):
[✓] UMN (~200MB) - from download_auto.py
[✓] Sample videos - from download_auto.py

Optional (Later):
[ ] UCY/ETH (~800MB)
[ ] MOT20 (~9GB)
[ ] Custom videos (~10-15GB)
```

---

## 💾 Storage Check

Run this to check available space:

```powershell
Get-PSDrive | Where-Object {$_.Name -eq (Get-Location).Drive.Name} | Select-Object Name, @{Name="Free(GB)";Expression={[math]::Round($_.Free/1GB,2)}}
```

**Required**: 60GB+ free space  
**Recommended**: 100GB+ (includes processing overhead)

---

## ⏱️ Estimated Download Times

| Connection Speed | Time for 26GB |
|-----------------|---------------|
| 10 Mbps | ~6 hours |
| 50 Mbps | ~1.5 hours |
| 100 Mbps | ~45 minutes |
| 500 Mbps | ~10 minutes |

**Tip**: Download overnight for slow connections!

---

## 🛠️ After Downloads Complete

Run preprocessing:
```powershell
python scripts\preprocess_datasets.py
```

This will:
- Filter COCO to person class only
- Convert annotations to unified format
- Generate train/val splits
- Create metadata files

---

## ❓ Troubleshooting

**Download interrupted?**
- Most zip managers can resume downloads
- Use `wget` or `aria2c` for resumable downloads

**Google Drive quota exceeded?**
- Try alternative links or Academic Torrents
- Wait 24 hours for quota reset

**MOT Challenge registration not working?**
- Use academic email if available
- Check spam folder for confirmation email

**Out of disk space?**
- Delete downloaded ZIP files after extraction
- Use external drive: change paths in download scripts

---

## 📧 Support

If links are broken or you need help:
1. Check DATASETS_GUIDE.md for alternative sources
2. Search dataset name on Papers With Code
3. Check Academic Torrents
4. Contact dataset authors (links in original papers)

---

**Ready to download? Run setup script first!**

```powershell
.\datasets\setup_datasets.ps1
```
