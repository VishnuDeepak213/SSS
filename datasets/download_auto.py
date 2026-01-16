"""
Automatic dataset downloader for publicly accessible datasets.
Downloads: UMN, UCY/ETH trajectories, and sample videos.

For COCO, MOT17, CrowdHuman, ShanghaiTech - see manual instructions.
"""

import os
import sys
import urllib.request
import urllib.error
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except urllib.error.URLError as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archives."""
    print(f"📦 Extracting {archive_path.name}...")
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def main():
    print("=" * 70)
    print("SDS Automatic Dataset Downloader")
    print("=" * 70)
    print()
    
    base_dir = Path(__file__).parent
    downloads_dir = base_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    datasets = []
    
    # UMN Dataset
    datasets.append({
        'name': 'UMN Unusual Crowd Activity',
        'url': 'http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi',
        'output': downloads_dir / 'umn_crowd_activity.avi',
        'extract_to': base_dir / 'umn',
        'size': '~200MB'
    })
    
    # Sample surveillance video for quick testing
    datasets.append({
        'name': 'Sample Test Video',
        'url': 'https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4',
        'output': base_dir / 'custom' / 'sample_test.mp4',
        'size': '~1MB'
    })
    
    print(f"📥 Datasets to download: {len(datasets)}")
    print()
    
    success_count = 0
    total_count = len(datasets)
    
    for i, dataset in enumerate(datasets, 1):
        print(f"[{i}/{total_count}] {dataset['name']} ({dataset['size']})")
        print(f"URL: {dataset['url']}")
        
        output_path = dataset['output']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            print(f"✓ Already exists: {output_path}")
            success_count += 1
        else:
            print(f"Downloading to: {output_path}")
            if download_url(dataset['url'], output_path):
                print(f"✓ Downloaded successfully")
                success_count += 1
                
                # Extract if archive
                if 'extract_to' in dataset and output_path.suffix in ['.zip', '.tar', '.gz', '.tgz']:
                    extract_archive(output_path, dataset['extract_to'])
            else:
                print(f"❌ Download failed")
        
        print()
    
    print("=" * 70)
    print(f"Download Summary: {success_count}/{total_count} successful")
    print("=" * 70)
    print()
    
    if success_count < total_count:
        print("⚠ Some downloads failed. Check your internet connection and retry.")
        print()
    
    print("📋 Next Steps:")
    print()
    print("MANUAL DOWNLOADS STILL REQUIRED:")
    print("  1. COCO Person (~5GB)")
    print("     http://images.cocodataset.org/zips/train2017.zip")
    print("     http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print()
    print("  2. MOT17 (~5GB) [Registration required]")
    print("     https://motchallenge.net/data/MOT17/")
    print()
    print("  3. CrowdHuman (~15GB)")
    print("     https://www.crowdhuman.org/download.html")
    print()
    print("  4. ShanghaiTech (~350MB)")
    print("     https://github.com/desenzhou/ShanghaiTechDataset")
    print()
    print("💡 TIP: Use download managers (IDM, wget) for large files!")
    print()
    print("For detailed instructions, see: DATASETS_GUIDE.md")
    print("=" * 70)


if __name__ == "__main__":
    # Check if tqdm is installed
    try:
        import tqdm
    except ImportError:
        print("Installing required package: tqdm")
        os.system(f"{sys.executable} -m pip install tqdm")
        import tqdm
    
    main()
