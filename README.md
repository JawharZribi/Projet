# 💡 Distributed Video Analysis System (Fog Computing Model)

This project implements a distributed video analysis pipeline using a **Host-Client (Fog Computing)** architecture. A central **Host** PC splits a large video file into smaller segments and distributes the workload to multiple **Client** PCs for simultaneous processing (vehicle detection and tracking). The Host then aggregates the results, creating a single merged video and a comprehensive traffic analysis report.

---

## ⚙️ Architecture Overview

The system operates as a Master-Worker paradigm across a local network (LAN):

1.  **Host (Master):** Splits the video using OpenCV, processes the first part locally, sends remaining parts to Clients via sockets, and aggregates all final results.
2.  **Clients (Workers):** Receive a video segment, execute the machine learning analysis (`safety_monitor.py`), and send the processed video and a local JSON report back to the Host.
3.  **`safety_monitor.py`:** Uses the **YOLOv8** model and **OpenCV** for vehicle detection and tracking, outputting bounding boxes, unique vehicle IDs, and confidence scores onto the video frames.



---

## ✨ Features

* **Distributed Processing:** Parallel video analysis across multiple nodes for faster execution.
* **Real Video Splitting:** Uses **OpenCV** (`cv2`) to perform frame-by-frame splitting, avoiding external dependencies like FFmpeg.
* **Socket-Based Communication:** Robust network communication for sending video parts and receiving results (Python's built-in `socket` module).
* **Object Tracking:** Tracks individual vehicles across frames, marking them with a unique ID and confidence score (e.g., `V-5: 98%`).
* **Data Aggregation:** Host automatically merges the video segments back into a single output file and combines all Client JSON reports into one final, comprehensive analysis report.

---

## 🚀 Getting Started

### Prerequisites

You must have **Python 3.x** installed on all machines (Host and Clients).

Install the required libraries on **ALL PCs**:

```bash
pip install opencv-python ultralytics numpy
