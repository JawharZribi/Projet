import sys
import os
import json
import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO

class SafetyMonitor:
    # COCO Class IDs for vehicles: 2=car, 3=motorcycle/van/pickup, 5=bus, 7=truck.
    VEHICLE_CLASSES = [2, 3, 5, 7] 

    def __init__(self, input_video_path, json_output_path, video_output_path):
        self.input_video_path = input_video_path
        self.json_output_path = json_output_path
        self.video_output_path = video_output_path
        self.model_loaded = False
        self.total_frames = 0
        self.total_vehicles_detected = 0
        self.next_vehicle_id = 1
        
        # Simple dictionary to store tracking info: {unique_id: [x1, y1, x2, y2]}
        self.tracking_data = {} 
        
        try:
            # Load a generic model trained on the COCO dataset
            self.model = YOLO("yolov8n.pt") 
            self.model_loaded = True
            print("Monitor: YOLO model loaded. Expecting vehicle detection.")
        except Exception as e:
            print(f"Monitor: Failed to load YOLO model: {e}")

    def get_iou(self, boxA, boxB):
        """Calculates Intersection over Union for tracking."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update_tracker(self, new_boxes, confidences):
        """
        Assigns a consistent ID to each detected vehicle based on proximity (simple IoU tracking).
        Returns a list of dictionaries: [{'box': box, 'id': id, 'conf': conf}]
        """
        tracked_objects = []
        used_ids = set()

        # 1. Match new detections to existing tracked objects
        for i, new_box in enumerate(new_boxes):
            max_iou = 0
            best_id = None
            
            for vehicle_id, old_box in self.tracking_data.items():
                iou = self.get_iou(old_box, new_box)
                if iou > max_iou and iou > 0.5: # 50% overlap required for match
                    max_iou = iou
                    best_id = vehicle_id
            
            if best_id and best_id not in used_ids:
                # Found a match
                tracked_objects.append({'box': new_box, 'id': best_id, 'conf': confidences[i]})
                self.tracking_data[best_id] = new_box # Update position
                used_ids.add(best_id)
            else:
                # New vehicle
                new_id = f"V-{self.next_vehicle_id}"
                self.tracking_data[new_id] = new_box
                tracked_objects.append({'box': new_box, 'id': new_id, 'conf': confidences[i]})
                self.next_vehicle_id += 1

        # 2. Cleanup (Simple: remove objects not detected in this frame) - *Simplified for demo*
        # A more complex tracker would keep the object for a few frames after it disappears.
        current_frame_ids = {obj['id'] for obj in tracked_objects}
        
        # We only keep objects whose bounding box was updated in this frame to keep the tracker lean
        self.tracking_data = {k: v for k, v in self.tracking_data.items() if k in current_frame_ids}
        
        return tracked_objects


    def process_frame(self, frame):
        """
        Detects specific vehicles using the YOLO model and overlays boxes and tracking IDs on the frame.
        """
        if not self.model_loaded:
             return frame, False, None
             
        # Detect all relevant vehicles
        results = self.model(frame, classes=self.VEHICLE_CLASSES, verbose=False)

        all_boxes = []
        all_confidences = []
        
        # Collect all detections
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            
            all_boxes.extend(boxes)
            all_confidences.extend(confidences)

        # Track and get IDs for each detection
        tracked_detections = self.update_tracker(all_boxes, all_confidences)
        
        is_detection_found = bool(tracked_detections)
        best_vehicle_box_data = None
        max_vehicle_area = 0
        
        for detection in tracked_detections:
            box = detection['box']
            vehicle_id = detection['id']
            conf = detection['conf']
            
            self.total_vehicles_detected += 1
            
            x1, y1, x2, y2 = box
            
            # Draw the bounding box (using a constant color for the task)
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create the label text: Vehicle ID and Confidence
            label = f"{vehicle_id}: {conf*100:.0f}%" # e.g., "V-5: 95%"
            
            # Put text above the bounding box
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
            
            # Check for the largest object to report as the primary anomaly
            area = (x2 - x1) * (y2 - y1)
            if area > max_vehicle_area:
                max_vehicle_area = area
                best_vehicle_box_data = {
                    'box': box, 
                    'id': vehicle_id, 
                    'conf': conf
                }
        
        anomaly_data = None
        if best_vehicle_box_data is not None:
            # Extract box data for JSON reporting
            box = best_vehicle_box_data['box']
            x = int(box[0])
            y = int(box[1])
            w = int(box[2] - box[0])
            h = int(box[3] - box[1])
            
            anomaly_data = {
                "type": "TRAFFIC_ANOMALY (Vehicle Tracked)", 
                "vehicle_id": best_vehicle_box_data['id'],
                "confidence": float(best_vehicle_box_data['conf']), 
                "reason": "Vehicle detected and tracked for traffic density and flow analysis.",
                "box": [x, y, w, h]
            }
                    
        return frame, is_detection_found, anomaly_data


    def run(self):
        """Main execution loop for video processing."""
        
        # Open the real video file
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print(f"Monitor: Error opening video file {self.input_video_path}")
            return

        # Setup VideoWriter
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0 or height == 0 or fps == 0:
            print(f"Monitor: Invalid video properties detected ({width}x{height}, {fps}fps). Skipping processing.")
            return

        out = cv2.VideoWriter(self.video_output_path, fourcc, fps, (width, height))

        final_report = {
            "analysis_result": "SUCCESS",
            "video_path": self.input_video_path,
            "total_frames": 0,
            "total_vehicles_detected": 0,
            "anomalies": []
        }

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process the frame
            processed_frame, has_detection, anomaly_data = self.process_frame(frame)
            
            if has_detection and anomaly_data is not None: 
                anomaly_data["frame"] = frame_count
                final_report["anomalies"].append(anomaly_data)
            
            # Write the frame with boxes to the output video
            out.write(processed_frame)

        cap.release()
        out.release()
        
        # Finalize report metrics
        final_report["total_frames"] = frame_count
        # Note: total_vehicles_detected is an approximation based on all detections per frame.
        final_report["total_vehicles_detected"] = self.total_vehicles_detected
        
        # Save the JSON report
        try:
            with open(self.json_output_path, 'w') as json_file:
                json.dump(final_report, json_file, indent=4)
            print(f"Monitor: Processing complete. Output JSON saved to {self.json_output_path}")
        except Exception as e:
            print(f"Monitor: Error saving JSON report: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python safety_monitor.py <input_video> <output_json> <output_video>")
        sys.exit(1)

    input_video, output_json, output_video = sys.argv[1], sys.argv[2], sys.argv[3]
    monitor = SafetyMonitor(input_video, output_json, output_video)
    monitor.run()