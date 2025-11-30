import cv2
import sys
import os
import json

def merge_videos_and_json(video_files_str, json_files_str, output_video_path, output_json_path):
    """Merges video segments and aggregates JSON reports."""
    
    # 1. Parse Input Paths
    video_files = video_files_str.split(',')
    json_files = json_files_str.split(',')
    
    # Ensure files exist and are in the correct order (part_0, part_1, part_2, etc.)
    video_files.sort() 
    json_files.sort()

    if not video_files:
        print("ERROR: No video files provided for merging.")
        return

    # 2. Get Properties from the first video
    cap = cv2.VideoCapture(video_files[0])
    if not cap.isOpened():
        print(f"ERROR: Cannot open first video segment: {video_files[0]}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 3. Setup Video Writer
    # Use 'mp4v' or 'XVID' codec for reliable MP4 output
    fourcc = cv2.VideoWriter.fourcc(*"mp4v") 
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out_video.isOpened():
        print(f"ERROR: Could not open VideoWriter for final output: {output_video_path}")
        return

    # 4. Merge Videos (Frame by Frame)
    print("UTILITY: Merging video frames...")
    cumulative_frame_count = 0
    
    for i, f_video in enumerate(video_files):
        cap_part = cv2.VideoCapture(f_video)
        if not cap_part.isOpened():
            print(f"WARNING: Skipping corrupted segment: {f_video}")
            continue

        while cap_part.isOpened():
            ret, frame = cap_part.read()
            if not ret:
                break
            
            # Ensure frame size matches the writer
            if frame.shape[1] != width or frame.shape[0] != height:
                 frame = cv2.resize(frame, (width, height))
            
            out_video.write(frame)
            cumulative_frame_count += 1
            
        cap_part.release()

    # 5. Release Video Resources
    out_video.release()
    print(f"UTILITY: Videos merged successfully. Total frames: {cumulative_frame_count}")

    # 6. Aggregate JSON Reports
    print("UTILITY: Aggregating JSON reports...")
    
    # Aggregate data needs to adjust frame numbers based on segment length
    total_frames_so_far = 0
    aggregated_alerts = []
    
    # NOTE: To get the true segment frame count, we need the original split info,
    # but for simplicity, we will append alerts and trust the host's logic
    # if total_frames_so_far is tracked properly.
    
    for f_json in json_files:
        try:
            with open(f_json, 'r') as f:
                data = json.load(f)
                
                # Check for the key used in safety_monitor.py
                if 'alerts' in data:
                    for alert in data['alerts']:
                        # Adjust frame number to reflect its position in the merged video
                        alert['frame'] += total_frames_so_far
                        aggregated_alerts.append(alert)
                    
                    # Estimate frames in this segment for the next segment's offset
                    total_frames_so_far += data.get('total_frames_processed', 0)
                
        except Exception as e:
            print(f"WARNING: Could not process JSON file {f_json}: {e}")
            
    final_report = {
        "total_segments_merged": len(video_files),
        "total_alerts": len(aggregated_alerts),
        "alerts": aggregated_alerts
    }
    
    with open(output_json_path, 'w') as json_file:
        json.dump(final_report, json_file, indent=4)
        
    print(f"UTILITY: JSON reports aggregated to {output_json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        # Host must pass video files, json files, output video, and output json path
        print("Usage: python merge_videos.py <comma_separated_video_files> <comma_separated_json_files> <output_video_path> <output_json_path>")
    else:
        merge_videos_and_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])