import sys # <-- ADDED
import socket
import threading
import subprocess
import os
import time
from typing import List, Optional
import cv2      # <--- REQUIRED FOR REAL SPLITTING
import math
import json  # <--- REQUIRED FOR JSON AGGREGATION


# --- CONFIGURATION ---
BASE_PORT = 5002
NUM_CLIENTS = 2
HOST_IP = '127.0.0.1' # Use loopback for local testing
VIDEO_PATH = "sample_video.mp4"
OUTPUT_MERGED_VIDEO = "output_merged.mp4"

# List to hold the file names of the processed parts (in order)
processed_parts: List[Optional[str]] = [None] * (NUM_CLIENTS + 1)

latch = threading.Semaphore(0) 
video_parts = []

# --- HELPER FUNCTIONS (Simulate Video Splitting and Merging) ---

def split_video(video_path, num_parts):
    """
    Splits the video into num_parts segments using OpenCV.
    """
    print(f"HOST: Splitting {video_path} into {num_parts} real parts...")
    
    if not os.path.exists(video_path):
        print(f"Error: Input video file not found at {video_path}")
        return [f"part_{i}.mp4" for i in range(num_parts)] # Fallback
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}. Check permissions or format.")
        return [f"part_{i}.mp4" for i in range(num_parts)]

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_part = total_frames // num_parts
    
    # Use 'mp4v' or 'XVID' codec for MP4 output
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 

    part_files = [f"part_{i}.mp4" for i in range(num_parts)]
    writers = []
    
    # 1. Initialize Video Writers
    for i in range(num_parts):
        writer = cv2.VideoWriter(part_files[i], fourcc, fps, (width, height))
        if not writer.isOpened():
             print(f"Error: Could not open writer for {part_files[i]}")
        writers.append(writer)

    # 2. Write frames to corresponding parts
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Determine which part this frame belongs to
        # Ensure the last part gets any remainder frames
        part_index = min(current_frame // frames_per_part, num_parts - 1)
        
        # Write frame
        if part_index < len(writers) and writers[part_index].isOpened():
            writers[part_index].write(frame)
        
        current_frame += 1

    # 3. Release resources
    cap.release()
    for writer in writers:
        if writer.isOpened():
            writer.release()

    print(f"HOST: Video split complete. Created files: {part_files}")
    return part_files

def merge_results(parts):
    """
    Merges processed video files and aggregates JSON reports using OpenCV.
    """
    video_files = [p for p in parts if p.endswith('.mp4')]
    json_files = [p.replace('.mp4', '.json') for p in parts if p.endswith('.mp4')]
    
    print("\nHOST: --- MERGING RESULTS ---")
    
    # --- 1. JSON Report Aggregation ---
    final_report = {
        "analysis_result": "SUCCESS",
        "total_frames": 0,
        "total_vehicles_detected": 0,
        "anomalies": []
    }
    
    print(f"HOST: Aggregating JSON reports: {json_files}")
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                report = json.load(f)
                final_report["total_frames"] += report.get("total_frames", 0)
                final_report["total_vehicles_detected"] += report.get("total_vehicles_detected", 0)
                # Concatenate anomaly lists
                final_report["anomalies"].extend(report.get("anomalies", []))
        except FileNotFoundError:
            print(f"HOST: WARNING: JSON report not found: {json_file}. Skipping.")
        except json.JSONDecodeError:
            print(f"HOST: ERROR: Failed to decode JSON report: {json_file}. Skipping.")

    # Save the aggregated JSON report
    aggregated_report_path = "aggregated_report.json"
    with open(aggregated_report_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"HOST: JSON Aggregation complete. Saved to {aggregated_report_path}")

    # --- 2. Video Merge ---
    
    if not video_files:
        print("HOST: WARNING: No video files found to merge.")
        return

    # Use properties of the first video part to set up the writer
    try:
        cap_ref = cv2.VideoCapture(video_files[0])
        if not cap_ref.isOpened():
             print(f"HOST: ERROR: Cannot open first video part {video_files[0]} for reference.")
             return

        fps = cap_ref.get(cv2.CAP_PROP_FPS)
        width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
        cap_ref.release()

        out = cv2.VideoWriter(OUTPUT_MERGED_VIDEO, fourcc, fps, (width, height))
    except Exception as e:
        print(f"HOST: ERROR setting up video writer: {e}")
        return

    # Iterate through all video parts and write frames sequentially
    print(f"HOST: Merging videos: {video_files} -> {OUTPUT_MERGED_VIDEO}")
    for i, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"HOST: WARNING: Cannot open {video_file}. Skipping this part.")
            continue
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()

    out.release()
    print("HOST: Merging and Aggregation complete.")
# --- FILE TRANSFER FUNCTIONS ---

def send_file(sock, file_path):
    """Sends a file over a socket, sending size first."""
    file_size = os.path.getsize(file_path)
    sock.sendall(str(file_size).encode().ljust(16))
    with open(file_path, 'rb') as f:
        while True:
            bytes_read = f.read(4096)
            if not bytes_read:
                break
            sock.sendall(bytes_read)
    print(f"HOST: Sent file: {file_path}")

def receive_file(sock, file_path):
    """Receives a file over a socket, reading size first."""
    file_size_str = sock.recv(16).decode().strip()
    if not file_size_str:
        return
    file_size = int(file_size_str)
    
    bytes_received = 0
    with open(file_path, 'wb') as f:
        while bytes_received < file_size:
            bytes_to_read = min(4096, file_size - bytes_received)
            chunk = sock.recv(bytes_to_read)
            if not chunk:
                break
            f.write(chunk)
            bytes_received += len(chunk)
    print(f"HOST: Received file: {file_path}")

# --- HOST PROCESSING THREADS ---

def client_handler(client_sock, client_id, video_part_path):
    """Handles the communication with a single Client Fog Node."""
    try:
        # 1. Send the video part to the client
        send_file(client_sock, video_part_path)
        client_sock.close() # Close initial connection

        # 2. Wait for the client to process and send results back
        result_port = BASE_PORT + client_id 
        result_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result_server_sock.bind((HOST_IP, result_port))
        result_server_sock.listen(1)
        print(f"HOST: Listening for result from Client {client_id} on port {result_port}...")
        
        result_sock, _ = result_server_sock.accept()
        print(f"HOST: Client {client_id} connected to send results.")
        
        # Define output file names
        output_video_part = f"host_output_part_{client_id}.mp4"
        
        # 3. Receive the processed video file
        receive_file(result_sock, output_video_part)
        
        # Store the received video part for merging
        processed_parts[client_id] = output_video_part 
        
    except Exception as e:
        print(f"HOST: Error handling client {client_id}: {e}")
    finally:
        if 'result_server_sock' in locals():
            result_server_sock.close()
        latch.release() # Signal that this thread is done

def local_processing(video_part_path):
    """Host processes its own video part (part 0) using the Safety Monitor logic."""
    try:
        output_video = "host_output_part_0.mp4"
        output_json = "host_output_report_0.json"
        
        print(f"\nHOST: Starting local processing on {video_part_path}...")
        
        # Execute the new Python processing script
        subprocess.run([
            sys.executable, "safety_monitor.py", # <-- FIXED: Using sys.executable
            video_part_path, 
            output_json, 
            output_video
        ], check=True)
        
        print(f"HOST: Local processing complete. Output saved to {output_video}")
        processed_parts[0] = output_video
        
    except subprocess.CalledProcessError as e:
        print(f"HOST: Local processing failed: {e}")
    except Exception as e:
        print(f"HOST: An error occurred during local processing: {e}")
    finally:
        latch.release() # Signal that local processing is done

# --- MAIN EXECUTION ---

def main():
    global video_parts
    
    # 1. Split the video (simulated)
    video_parts = split_video(VIDEO_PATH, NUM_CLIENTS + 1)
    
    # 2. Setup Server Socket to accept initial client connections
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST_IP, BASE_PORT))
    server_sock.listen(NUM_CLIENTS)
    print(f"HOST: Server listening on {HOST_IP}:{BASE_PORT} for {NUM_CLIENTS} clients...")

    client_threads = []
    
    # 3. Accept all client connections and assign tasks
    for i in range(1, NUM_CLIENTS + 1):
        try:
            # Java Host created separate ports for initial connection (5003, 5004, etc.), 
            # but we use a single initial port (5002) for the modern Python design.
            client_sock, addr = server_sock.accept()
            print(f"HOST: Accepted connection from Client {i} at {addr}")
            
            # Start client handler thread
            t = threading.Thread(target=client_handler, args=(client_sock, i, video_parts[i]))
            client_threads.append(t)
            t.start()
        except socket.error as e:
            print(f"HOST: Socket error during client connection: {e}")
            break
            
    server_sock.close() # Close initial listening socket

    # 4. Start local processing in a separate thread (part 0)
    local_thread = threading.Thread(target=local_processing, args=(video_parts[0],))
    local_thread.start()
    client_threads.append(local_thread)

    # 5. Wait for all processing threads to finish
    print(f"\nHOST: Waiting for all {NUM_CLIENTS + 1} processing tasks to complete...")
    for _ in range(NUM_CLIENTS + 1):
        latch.acquire() 

    # 6. Merge results
    if all(processed_parts):
        merge_results(processed_parts)
    else:
        print("\nHOST: WARNING: Not all parts were successfully processed. Skipping merge.")

    print("HOST: Process completed.")

if __name__ == "__main__":
    main()