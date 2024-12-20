import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from pathlib import Path
import argparse

class ExerciseTracker:
    def __init__(self, model_path='/home/kranti/Documents/yolov7-object-tracking-main/yolov8n-pose.pt'):  
        # Initialize YOLOv8 pose model
        self.model = YOLO(model_path)
        
        # Exercise state variables
        self.counter = 0
        self.stage = None  
        self.position = None
        
        # Configurable parameters
        self.min_push_up_angle = 80
        self.max_push_up_angle = 160
        
        # Store angles for form analysis
        self.angles_history = {'right': [], 'left': []}
        
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        if None in (point1, point2, point3):
            return None
            
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
        
    def process_frame(self, frame):
        """Process a single frame and return the annotated frame"""
        results = self.model(frame, verbose=False)
        
        if len(results) == 0:
            return frame, "No person detected"
            
        keypoints = results[0].keypoints.data[0] if len(results[0].keypoints) > 0 else None
        
        if keypoints is None:
            return frame, "No keypoints detected"
            
        try:
            right_shoulder = (int(keypoints[5][0]), int(keypoints[5][1]))
            right_elbow = (int(keypoints[7][0]), int(keypoints[7][1]))
            right_wrist = (int(keypoints[9][0]), int(keypoints[9][1]))
            
            left_shoulder = (int(keypoints[6][0]), int(keypoints[6][1]))
            left_elbow = (int(keypoints[8][0]), int(keypoints[8][1]))
            left_wrist = (int(keypoints[10][0]), int(keypoints[10][1]))
            
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if right_angle is not None:
                self.angles_history['right'].append(right_angle)
                if len(self.angles_history['right']) > 10:
                    self.angles_history['right'].pop(0)
                
                if right_angle > self.max_push_up_angle:
                    self.position = "up"
                elif right_angle < self.min_push_up_angle and self.position == "up":
                    self.position = "down"
                    self.counter += 1
                
                cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
                cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)
                cv2.putText(frame, f'Right Angle: {int(right_angle)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if left_angle is not None:
                self.angles_history['left'].append(left_angle)
                if len(self.angles_history['left']) > 10:
                    self.angles_history['left'].pop(0)
                
                cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_elbow, 5, (0, 255, 0), -1)
                cv2.circle(frame, left_wrist, 5, (0, 255, 0), -1)
                cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 2)
                cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 2)
                cv2.putText(frame, f'Left Angle: {int(left_angle)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(frame, f'Counter: {self.counter}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
        except Exception as e:
            print(f"Error processing keypoints: {e}")
            
        return frame, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source (0 for webcam)')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt', help='model path')
    parser.add_argument('--output', type=str, default='output.avi', help='/home/kranti/Documents/yolov7-object-tracking-main/runs/detect')  # Added output argument
    args = parser.parse_args()
    
    tracker = ExerciseTracker(model_path=args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isnumeric() else args.source)
    
    # Initialize VideoWriter
    target_width, target_height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
    out = cv2.VideoWriter(args.output, fourcc, 30, (target_width, target_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, error = tracker.process_frame(frame)
        annotated_frame = cv2.resize(annotated_frame, (target_width, target_height))
        
        if error:
            cv2.putText(annotated_frame, error, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(annotated_frame)  # Write frame to output video file
        
        cv2.imshow("Video Output", annotated_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
