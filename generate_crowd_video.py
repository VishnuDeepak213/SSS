"""Generate synthetic high-density crowd video for testing."""
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_synthetic_crowd_video(output_path, duration_seconds=10, fps=30, width=1280, height=720):
    """Create a synthetic crowd video with persons moving."""
    logger_print = lambda *args: print("[SyntheticCrowd]", *args)
    
    logger_print(f"Creating synthetic crowd video...")
    logger_print(f"Duration: {duration_seconds}s, FPS: {fps}, Resolution: {width}x{height}")
    
    total_frames = duration_seconds * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Create persons (x, y, vx, vy, size, color)
    np.random.seed(42)
    num_persons = 15  # Start with 15 persons
    persons = []
    
    for i in range(num_persons):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-3, 3)
        size = np.random.randint(40, 80)
        color = tuple(np.random.randint(100, 256, 3).tolist())
        persons.append([x, y, vx, vy, size, color])
    
    for frame_idx in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Increase density over time (crowding effect)
        if frame_idx > total_frames * 0.3:  # After 30% of video
            if len(persons) < 25:
                for _ in range(2):
                    x = np.random.randint(50, width - 50)
                    y = np.random.randint(50, height - 50)
                    vx = np.random.uniform(-2, 2)
                    vy = np.random.uniform(-2, 2)
                    size = np.random.randint(30, 70)
                    color = tuple(np.random.randint(100, 256, 3).tolist())
                    persons.append([x, y, vx, vy, size, color])
        
        # Create panic/anomaly effect at frame 150-250
        anomaly_active = 150 <= frame_idx < 250
        
        # Update and draw persons
        for i, person in enumerate(persons):
            x, y, vx, vy, size, color = person
            
            # Add panic effect
            if anomaly_active and i % 3 == 0:
                vx *= 2.5  # Speed up
                vy *= 2.5
            
            # Update position
            x += vx
            y += vy
            
            # Collision detection / crowd interaction
            if x < 0 or x > width:
                vx *= -1
                x = np.clip(x, 0, width)
            if y < 0 or y > height:
                vy *= -1
                y = np.clip(y, 0, height)
            
            # Random direction changes occasionally
            if frame_idx % 60 == 0 and np.random.random() < 0.3:
                vx += np.random.uniform(-1, 1)
                vy += np.random.uniform(-1, 1)
            
            persons[i] = [x, y, vx, vy, size, color]
            
            # Draw person as circle
            cv2.circle(frame, (int(x), int(y)), int(size//2), color, -1)
            cv2.circle(frame, (int(x), int(y)), int(size//2), (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Crowd Size: {len(persons)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Add anomaly indicator
        if anomaly_active:
            cv2.rectangle(frame, (20, 110), (400, 150), (0, 0, 255), -1)
            cv2.putText(frame, "ANOMALY: Panic/Running Detected!", (30, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate and show crowd density
        density = len(persons)
        if density >= 20:
            density_text = "CRITICAL"
            density_color = (0, 0, 255)
        elif density >= 15:
            density_text = "HIGH"
            density_color = (0, 165, 255)
        elif density >= 10:
            density_text = "MEDIUM"
            density_color = (0, 255, 255)
        else:
            density_text = "LOW"
            density_color = (0, 255, 0)
        
        cv2.rectangle(frame, (width-250, 20), (width-10, 100), density_color, -1)
        cv2.putText(frame, f"Density: {density_text}", (width-240, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Persons: {density}", (width-240, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        if (frame_idx + 1) % 60 == 0:
            logger_print(f"Frame {frame_idx+1}/{total_frames} - Crowd: {len(persons)}")
    
    out.release()
    logger_print(f"Synthetic video created: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic crowd video")
    parser.add_argument("--output", type=str, default="test_crowd_video.mp4",
                       help="Output video path")
    parser.add_argument("--duration", type=int, default=10,
                       help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    create_synthetic_crowd_video(output_path, args.duration, args.fps)
    print(f"\nVideo ready for analysis: {output_path}")


if __name__ == "__main__":
    main()
