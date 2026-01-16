# SDS Visualization Test Results

## Summary
Successfully tested the SDS (Smart Detection & Surveillance) system with crowd density estimation, tracking, and visualization features.

## Test Images Processed

### 1. Bus Image (3 persons detected)
- **Input**: `venv/Lib/site-packages/ultralytics/assets/bus.jpg`
- **Size**: 810x1080 pixels
- **Detections**: 3 persons
- **Density Level**: LOW

**Generated Outputs**:
1. `bus_01_detections.jpg` - Basic person detection with bounding boxes and confidence scores
2. `bus_02_tracking.jpg` - Tracking visualization with track IDs
3. `bus_03_density_heatmap.jpg` - Crowd density heatmap with grid counts
4. `bus_04_full_visualization.jpg` - Complete visualization with all features

### 2. Zidane Image (2 persons detected)
- **Input**: `venv/Lib/site-packages/ultralytics/assets/zidane.jpg`
- **Size**: 1280x720 pixels
- **Detections**: 2 persons
- **Density Level**: LOW

**Generated Outputs**:
1. `zidane_01_detections.jpg` - Basic person detection with bounding boxes and confidence scores
2. `zidane_02_tracking.jpg` - Tracking visualization with track IDs
3. `zidane_03_density_heatmap.jpg` - Crowd density heatmap with grid counts
4. `zidane_04_full_visualization.jpg` - Complete visualization with all features

## Features Demonstrated

### Person Detection
- YOLOv8 nano model
- Confidence threshold: 0.5
- Person class detection (COCO class 0)
- Bounding boxes with confidence scores

### Tracking
- DeepSORT multi-person tracker
- Track ID assignment
- Unique color per track
- Track history maintenance

### Crowd Density Estimation
- Grid-based density analysis (8x6 grid)
- Heatmap visualization with JET colormap
- Per-cell person counts
- Density level classification:
  - LOW: < 5 persons
  - MEDIUM: 5-15 persons
  - HIGH: 15-30 persons
  - CRITICAL: > 30 persons

### Visualization Outputs
1. **Detections**: Green bounding boxes with confidence scores
2. **Tracking**: Color-coded tracks with IDs
3. **Density Heatmap**: Blended heatmap overlay showing crowd concentration
4. **Full Visualization**: Combined view of all features

## How to View Results

### Option 1: Manual
Navigate to the test_results folder:
```
test_results/
├── bus/
│   ├── bus_01_detections.jpg
│   ├── bus_02_tracking.jpg
│   ├── bus_03_density_heatmap.jpg
│   └── bus_04_full_visualization.jpg
└── zidane/
    ├── zidane_01_detections.jpg
    ├── zidane_02_tracking.jpg
    ├── zidane_03_density_heatmap.jpg
    └── zidane_04_full_visualization.jpg
```

### Option 2: Using the Viewer Script
```bash
python show_results.py
```
This will display all images sequentially. Press any key to advance, ESC to exit.

### Option 3: File Explorer
```bash
explorer test_results
```

## Testing Your Own Images/Videos

### Test a Single Image
```bash
python test_visualization.py --input path/to/image.jpg --output test_results/custom
```

### Test a Video
```bash
python test_visualization.py --input path/to/video.mp4 --output test_results/video --max-frames 300
```

### Batch Testing
Edit `run_tests.py` to add your images/videos and run:
```bash
python run_tests.py
```

## System Configuration

**Detection Model**: YOLOv8 nano (CPU mode)
**Device**: CPU (configured in config/config.yaml)
**Grid Size**: 8x6
**Processing Speed**: ~8-10 FPS on CPU

## Notes

- For single images, tracking shows 0 active tracks since tracks require multiple frames to confirm
- Density estimation works on both single images (using detections) and videos (using tracks)
- The heatmap uses a JET colormap where:
  - Blue = Low density
  - Green/Yellow = Medium density
  - Red = High density
- All visualizations are saved as JPEG files for easy viewing

## Next Steps

To test with real crowd scenes:
1. Download crowd videos or images
2. Use `test_visualization.py` to process them
3. Compare density levels and tracking performance
4. Adjust thresholds in `config/config.yaml` as needed
