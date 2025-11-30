import socket
import subprocess
import os
import sys

# --- CONFIGURATION ---
HOST_IP = '127.0.0.1' # Must match host.py
INITIAL_PORT = 5002 # Host's initial listening port
CLIENT_ID = 1      # Default Client ID (launch multiple instances with different IDs)

# --- COMMUNICATION FUNCTIONS ---

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
    print(f"CLIENT {CLIENT_ID}: Sent file: {file_path}")

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
    print(f"CLIENT {CLIENT_ID}: Received file: {file_path}")

# --- PROCESSING FUNCTION ---

def process_video_part(video_path, json_output, video_output):
    """Executes the external Python Safety Monitor script."""
    try:
        print(f"CLIENT {CLIENT_ID}: Starting video processing (Traffic monitoring)...") # CHANGED MESSAGE
        # Execute the core logic script
        subprocess.run([
            sys.executable, "safety_monitor.py", # <-- FIXED: Using sys.executable
            video_path, 
            json_output, 
            video_output
        ], check=True)
        
        print(f"CLIENT {CLIENT_ID}: Processing complete. Output saved to {video_output}")
        
    except subprocess.CalledProcessError as e:
        print(f"CLIENT {CLIENT_ID}: Processing failed with error: {e}")
    except Exception as e:
        print(f"CLIENT {CLIENT_ID}: An error occurred during processing: {e}")

# --- MAIN EXECUTION ---

def main():
    global CLIENT_ID
    
    # Allow passing CLIENT_ID as argument (e.g., python client.py 2)
    if len(sys.argv) > 1:
        try:
            CLIENT_ID = int(sys.argv[1])
        except ValueError:
            print(f"Invalid client ID: {sys.argv[1]}. Using default {CLIENT_ID}.")
    
    # The Client result port is offset by its ID (5002 + 1 = 5003, 5002 + 2 = 5004, etc.)
    RESULT_PORT = INITIAL_PORT + CLIENT_ID 
    
    try:
        # 1. Connect to Host on initial port to receive the task (video part)
        print(f"CLIENT {CLIENT_ID}: Connecting to Host at {HOST_IP}:{INITIAL_PORT} to receive task...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST_IP, INITIAL_PORT))
            print(f"CLIENT {CLIENT_ID}: Connected. Receiving video part.")

            # Define input file name
            received_video_part = f"received_part_{CLIENT_ID}.mp4"
            receive_file(sock, received_video_part)
            
        # 2. Process the received video part
        json_output = f"client_report_{CLIENT_ID}.json"
        video_output = f"client_output_{CLIENT_ID}.mp4"
        process_video_part(received_video_part, json_output, video_output)

        # 3. Connect back to Host on the result port to send results
        print(f"CLIENT {CLIENT_ID}: Connecting to Host at {HOST_IP}:{RESULT_PORT} to send results...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_sock:
            result_sock.connect((HOST_IP, RESULT_PORT))
            
            # Send the processed video file
            send_file(result_sock, video_output)
            # You can optionally send the JSON report here too
            # send_file(result_sock, json_output)

            print(f"CLIENT {CLIENT_ID}: Results sent back to Host.")
            
    except ConnectionRefusedError:
        print(f"CLIENT {CLIENT_ID}: Connection refused. Ensure Host is running and listening.")
    except Exception as e:
        print(f"CLIENT {CLIENT_ID}: An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Launch multiple clients by passing the ID as an argument:
    # python client.py 1
    # python client.py 2
    main()