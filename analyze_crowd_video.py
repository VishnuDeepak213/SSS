"""Advanced crowd video analysis with density, flow, tracking, and anomaly detection."""
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
from collections import deque

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.density.estimator import DensityEstimator
from src.flow.analyzer import FlowAnalyzer
from src.threats.detector import ThreatDetector
from src.visualization.renderer import Visualizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class CrowdVideoAnalyzer:
    """Advanced crowd video analysis with multiple features."""
    
    def __init__(self, config):
        self.config = config
        self.detector = PersonDetector(config['detection'])
        self.tracker = PersonTracker(config['tracking'])
        self.density_estimator = None
        self.flow_analyzer = FlowAnalyzer(config['flow'])
        self.threat_detector = ThreatDetector(config['threats'])
        self.visualizer = Visualizer(config['visualization'])
        
        # Track history for anomaly detection
        self.track_velocities = {}
        self.track_positions = {}
        self.anomalies = []
        self.frame_count = 0
        
    def init_for_video(self, w, h):
        """Initialize video-specific components."""
        self.density_estimator = DensityEstimator(self.config['density'], (w, h))
        self.frame_width = w
        self.frame_height = h
    
    def detect_anomalies(self, frame, tracks):
        """Detect anomalies: rapid movements, isolated persons, etc."""
        anomalies = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = int(track.track_id)
            bbox = track.to_ltrb()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Check for rapid movement (velocity spike)
            if track_id in self.track_positions:
                prev_pos = self.track_positions[track_id]
                distance = np.sqrt((center_x - prev_pos[0])**2 + (center_y - prev_pos[1])**2)
                velocity = distance / (self.config['flow']['flow_interval'] or 1)
                
                # Anomaly: Very fast movement (potential running/panic)
                if velocity > 50:  # pixels per frame
                    anomalies.append({
                        'type': 'rapid_movement',
                        'track_id': track_id,
                        'velocity': velocity,
                        'position': (int(center_x), int(center_y)),
                        'severity': 'HIGH' if velocity > 80 else 'MEDIUM'
                    })
                
                self.track_velocities[track_id] = velocity
            
            self.track_positions[track_id] = (center_x, center_y)
        
        return anomalies
    
    def process_frame(self, frame, frame_idx):
        """Process a single frame and return visualization."""
        self.frame_count = frame_idx
        
        # Detect persons
        detections = self.detector(frame)
        
        # Track persons
        tracks = self.tracker.update(frame, detections)
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        
        # Estimate density
        density_grid, density_heatmap, density_alerts = self.density_estimator.estimate(confirmed_tracks)
        total_count = density_grid.sum()
        
        # Determine density level
        if total_count >= self.config['density']['thresholds']['critical']:
            density_level = 'CRITICAL'
            density_color = (0, 0, 255)  # Red
        elif total_count >= self.config['density']['thresholds']['high']:
            density_level = 'HIGH'
            density_color = (0, 165, 255)  # Orange
        elif total_count >= self.config['density']['thresholds']['medium']:
            density_level = 'MEDIUM'
            density_color = (0, 255, 255)  # Yellow
        else:
            density_level = 'LOW'
            density_color = (0, 255, 0)  # Green
        
        # Analyze optical flow
        flow_result = self.flow_analyzer.analyze(frame)
        if flow_result[0] is not None:
            flow, flow_viz, flow_stats = flow_result
        else:
            flow, flow_viz, flow_stats = None, None, None
        
        # Detect anomalies
        anomalies = self.detect_anomalies(frame, confirmed_tracks)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw density heatmap
        if density_heatmap is not None and self.config['visualization']['show_density_heatmap']:
            if density_heatmap.shape[:2] != frame.shape[:2]:
                density_heatmap = cv2.resize(density_heatmap, 
                                           (self.frame_width, self.frame_height))
            vis_frame = cv2.addWeighted(vis_frame, 0.7, density_heatmap, 0.3, 0)
        
        # Draw flow vectors
        if flow_viz is not None and self.config['visualization']['show_flow_arrows']:
            if flow_viz.shape[:2] == frame.shape[:2]:
                vis_frame = cv2.addWeighted(vis_frame, 0.8, flow_viz, 0.2, 0)
        
        # Draw bounding boxes and tracks
        if self.config['visualization']['show_bboxes']:
            for track in confirmed_tracks:
                bbox = track.to_ltrb()
                x1, y1, x2, y2 = map(int, bbox)
                track_id = int(track.track_id)
                
                # Color based on anomaly status
                color = (0, 255, 0)  # Default green
                for anomaly in anomalies:
                    if anomaly['track_id'] == track_id:
                        color = (0, 0, 255) if anomaly['severity'] == 'HIGH' else (0, 165, 255)
                        break
                
                # Draw box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and velocity
                label = f"ID:{track_id}"
                if track_id in self.track_velocities:
                    vel = self.track_velocities[track_id]
                    label += f" V:{vel:.1f}"
                
                cv2.putText(vis_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw density stats
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Tracking: {len(confirmed_tracks)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw density info with color
        cv2.rectangle(vis_frame, (10, 140), (280, 200), density_color, -1)
        cv2.putText(vis_frame, f"Crowd Density: {density_level}", (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Count: {total_count} persons", (20, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw anomalies
        anomaly_count = len(anomalies)
        if anomaly_count > 0:
            cv2.rectangle(vis_frame, (10, 210), (350, 260), (0, 0, 255), -1)
            cv2.putText(vis_frame, f"ANOMALIES DETECTED: {anomaly_count}", (20, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            for i, anomaly in enumerate(anomalies[:3]):  # Show top 3
                y_offset = 270 + i * 30
                if y_offset < vis_frame.shape[0] - 10:
                    text = f"ID:{anomaly['track_id']} - {anomaly['type']}"
                    cv2.putText(vis_frame, text, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Draw density grid overlay
        if self.config['visualization']['show_grid']:
            grid_h, grid_w = density_grid.shape
            cell_h = self.frame_height // grid_h
            cell_w = self.frame_width // grid_w
            
            for i in range(grid_h):
                for j in range(grid_w):
                    x1 = j * cell_w
                    y1 = i * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    count = density_grid[i, j]
                    
                    if count > 0:
                        # Grid cell color based on density
                        if count >= self.config['density']['thresholds']['high']:
                            color = (0, 0, 200)
                        elif count >= self.config['density']['thresholds']['medium']:
                            color = (0, 200, 200)
                        else:
                            color = (200, 200, 0)
                        
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)
                        # Draw count
                        cv2.putText(vis_frame, str(count), 
                                   (x1 + cell_w//3, y1 + cell_h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame, {
            'frame': frame_idx,
            'detections': len(detections),
            'tracks': len(confirmed_tracks),
            'density': total_count,
            'density_level': density_level,
            'anomalies': anomaly_count,
            'anomaly_details': anomalies
        }


def process_video(video_path, config, output_dir):
    """Process video with complete analysis."""
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Initialize analyzer
    analyzer = CrowdVideoAnalyzer(config)
    analyzer.init_for_video(w, h)
    
    # Output video writer
    video_name = Path(video_path).stem
    output_video = output_dir / f"{video_name}_analyzed.mp4"
    stats_file = output_dir / f"{video_name}_analysis.txt"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    
    frame_idx = 0
    stats = []
    
    logger.info("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, frame_stats = analyzer.process_frame(frame, frame_idx)
        
        # Write frame
        out.write(vis_frame)
        stats.append(frame_stats)
        
        # Log progress
        if frame_idx % 30 == 0:
            logger.info(f"Frame {frame_idx}/{total_frames}: "
                       f"{frame_stats['detections']} det, "
                       f"{frame_stats['tracks']} tracks, "
                       f"Density: {frame_stats['density_level']}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Save statistics
    with open(stats_file, 'w') as f:
        f.write("Video Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Resolution: {w}x{h}\n")
        f.write(f"FPS: {fps:.1f}\n")
        f.write(f"Total Frames: {frame_idx}\n\n")
        
        # Overall statistics
        if stats:
            avg_detections = np.mean([s['detections'] for s in stats])
            avg_tracks = np.mean([s['tracks'] for s in stats])
            avg_density = np.mean([s['density'] for s in stats])
            max_density = max([s['density'] for s in stats])
            total_anomalies = sum([s['anomalies'] for s in stats])
            
            f.write("Overall Statistics\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average Detections per Frame: {avg_detections:.1f}\n")
            f.write(f"Average Active Tracks: {avg_tracks:.1f}\n")
            f.write(f"Average Crowd Density: {avg_density:.1f} persons\n")
            f.write(f"Maximum Crowd Density: {max_density} persons\n")
            f.write(f"Total Anomalies Detected: {total_anomalies}\n\n")
            
            # Density distribution
            f.write("Density Distribution\n")
            f.write("-" * 70 + "\n")
            for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                count = sum(1 for s in stats if s['density_level'] == level)
                percentage = (count / len(stats)) * 100
                f.write(f"{level}: {count} frames ({percentage:.1f}%)\n")
            
            # Anomalies summary
            if total_anomalies > 0:
                f.write(f"\nAnomalies Summary\n")
                f.write("-" * 70 + "\n")
                anomaly_types = {}
                for s in stats:
                    for anomaly in s['anomaly_details']:
                        atype = anomaly['type']
                        anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
                
                for atype, count in anomaly_types.items():
                    f.write(f"{atype}: {count} occurrences\n")
    
    logger.info(f"Processed {frame_idx} frames")
    logger.info(f"Output video: {output_video}")
    logger.info(f"Analysis report: {stats_file}")
    
    return output_video, stats_file


def main():
    parser = argparse.ArgumentParser(description="Advanced crowd video analysis")
    parser.add_argument("--input", type=str, required=True,
                       help="Input video file")
    parser.add_argument("--output", type=str, default="crowd_analysis",
                       help="Output directory")
    args = parser.parse_args()
    
    # Setup
    config = load_config()
    config['visualization']['show_density_heatmap'] = True
    config['visualization']['show_flow_arrows'] = True
    config['visualization']['show_bboxes'] = True
    config['visualization']['show_grid'] = True
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info("=" * 70)
    logger.info("ADVANCED CROWD VIDEO ANALYSIS")
    logger.info("Features: Detection, Tracking, Density, Flow, Anomaly Detection")
    logger.info("=" * 70)
    
    # Process video
    output_video, stats_file = process_video(input_path, config, output_dir)
    
    if output_video:
        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"Output video: {output_video}")
        logger.info(f"Statistics: {stats_file}")
        logger.info("=" * 70)
    else:
        logger.error("Video processing failed")


if __name__ == "__main__":
    main()
