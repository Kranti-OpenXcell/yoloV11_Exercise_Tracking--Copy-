import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import argparse

class ExerciseTracker:
    def __init__(self, model_path='/home/kranti/Documents/yolov7-object-tracking-main/yolov8n-pose.pt'):  
        # Initialize YOLOv8 pose model
        self.model = YOLO(model_path)
        
        # Exercise state variables
        self.counters = {}  # For multiple people, each person's counter
        self.positions = {}  # For multiple people, each person's position
        self.direction = {}  # For tracking the direction (up/down) of each person
        self.reps = {}  # Rep count for each person

        # Configurable parameters
        self.min_push_up_angle = 80
        self.max_push_up_angle = 160
        self.min_hip_angle = 140  # Angle at hip for push-ups
        self.max_hip_angle = 180  # Angle at hip for push-ups
        
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
        
        people = results[0].keypoints.data  # Keypoints for all detected people
        if len(people) == 0:
            return frame, "No keypoints detected"
        
        try:
            for i, keypoints in enumerate(people):
                # Each person has their own set of keypoints
                right_shoulder = (int(keypoints[5][0]), int(keypoints[5][1]))
                right_elbow = (int(keypoints[7][0]), int(keypoints[7][1]))
                right_wrist = (int(keypoints[9][0]), int(keypoints[9][1]))
                left_shoulder = (int(keypoints[6][0]), int(keypoints[6][1]))
                left_elbow = (int(keypoints[8][0]), int(keypoints[8][1]))
                left_wrist = (int(keypoints[10][0]), int(keypoints[10][1]))
                
                # Adding hip and ankle tracking
                right_hip = (int(keypoints[11][0]), int(keypoints[11][1]))
                left_hip = (int(keypoints[12][0]), int(keypoints[12][1]))
                
                # Calculate angles for arms and hips
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_elbow)
                left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_elbow)
                right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_wrist)
                left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_wrist)

                
                # Initialize rep count for new person
                if i not in self.reps:
                    self.reps[i] = 0
                    self.direction[i] = 0  # 0 = down, 1 = up
                
                # Rep count logic: detect the angle transitions for push-up
                if right_elbow_angle and left_elbow_angle:
                    # if right_elbow_angle <= self.min_push_up_angle and left_elbow_angle <= self.min_push_up_angle and self.direction[i] == 0:
                    #     # Going up
                    #     self.reps[i] += 1
                    #     self.direction[i] = 1
                    # elif right_elbow_angle >= self.max_push_up_angle and left_elbow_angle >= self.max_push_up_angle and self.direction[i] == 1:
                    #     # Going down
                    #     self.direction[i] = 0

                 # Going down condition
                   
                   
                    if (right_elbow_angle <= 90 and right_shoulder_angle < 20):
                            # feedback = "Feedback: Go Up"
                            # if direction == 0:  # Only increment count and update direction when moving down
                        self.reps[i] += 0.5
                        self.direction[i] = 1
                    else:
                        feedback = "Feedback: Maintain Form."

                    # Going up condition
                    if (right_elbow_angle > 160 and right_shoulder_angle > 30):
                        # Feedback: "Going up"
                        # if direction == 1:  # Only increment count when moving up (and previously going down)
                        self.reps[i] += 0.5
                        self.direction[i] = 1
                                    
                # Show angles at the joint location (elbow/hip)
                if right_elbow_angle is not None:
                    cv2.putText(frame, f'{int(right_elbow_angle)}', right_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_elbow_angle is not None:
                    cv2.putText(frame, f'{int(left_elbow_angle)}', left_elbow, cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 2)
                if right_hip_angle is not None:
                    cv2.putText(frame, f'{int(right_hip_angle)}', right_hip, cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 2)
                if left_hip_angle is not None:
                    cv2.putText(frame, f'{int(left_hip_angle)}', left_hip, cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 2)
                if right_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(right_shoulder_angle)}', right_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(left_shoulder_angle)}',left_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Display Rep count above each person
                cv2.putText(frame, f'Reps: {self.reps[i]}', (right_shoulder[0] - 30, right_shoulder[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Optionally: Draw circles and lines for keypoints and angles for better visualization
                cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
                cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)
                
                # For the hip angle visualization (lines for hips)
                cv2.circle(frame, right_hip, 5, (0, 255, 0), -1)
                cv2.line(frame, right_shoulder, right_hip, (0, 255, 0), 2)

                cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, left_shoulder, left_elbow, (255, 0, 0), 2)
                cv2.line(frame, left_elbow, left_wrist, (255, 0, 0), 2)
                
                # For the hip angle visualization (lines for hips)
                cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)
                cv2.line(frame, left_shoulder, left_hip, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Error processing keypoints: {e}")
            
        return frame, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source (0 for webcam)')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt', help='model path')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video file')  # Output path
    args = parser.parse_args()
    
    tracker = ExerciseTracker(model_path=args.model)
    cap = cv2.VideoCapture(int(args.source) if args.source.isnumeric() else args.source)
    
    # # Initialize VideoWriter
    target_width, target_height = 1280, 720
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    # out = cv2.VideoWriter(args.output, fourcc, 30, (target_width, target_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, error = tracker.process_frame(frame)
        annotated_frame = cv2.resize(annotated_frame, (target_width, target_height))
        # out.write(annotated_frame)
        
        cv2.imshow("Push-Up Tracker", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
