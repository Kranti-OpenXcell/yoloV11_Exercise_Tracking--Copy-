import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import csv
import argparse

class ExerciseTracker:
    def __init__(self, model_path='/home/kranti/Documents/yoloV11_Exercise_Tracking/yolo11n-pose.pt'):  
        self.model = YOLO(model_path)
        self.people_data = {}
        self.last_positions = {}
        self.frames_since_seen = {}
        self.id_counter = 0
        self.max_distance_threshold = 100
        self.max_frames_missing = 30
        self.frame_log = []
        self.start_time = time.time()

    def get_person_center(self, keypoints):
        valid_points = []
        for point_idx in [5, 6, 11, 12]:
            if not (keypoints[point_idx][0].item() == 0 and keypoints[point_idx][1].item() == 0):
                valid_points.append((keypoints[point_idx][0].item(), keypoints[point_idx][1].item()))
        
        if not valid_points:
            return None
            
        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)
        return (center_x, center_y)

    def assign_id(self, center):
        min_distance = float('inf')
        closest_id = None
        for person_id, last_pos in self.last_positions.items():
            if self.frames_since_seen[person_id] < self.max_frames_missing:
                distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                if distance < min_distance and distance < self.max_distance_threshold:
                    min_distance = distance
                    closest_id = person_id
        
        if closest_id is None:
            closest_id = self.id_counter
            self.id_counter += 1
            self.people_data[closest_id] = {'reps': 0, 'position': 'up'}
        
        self.last_positions[closest_id] = center
        self.frames_since_seen[closest_id] = 0
        return closest_id

    def calculate_angle(self, point1, point2, point3):
        if None in (point1, point2, point3):
            return None
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame, frame_number):
        results = self.model(frame, verbose=False)
        if len(results) == 0:
            return frame, "No person detected"
        
        people = results[0].keypoints.data
        if len(people) == 0:
            return frame, "No keypoints detected"

        for person_id in self.frames_since_seen:
            self.frames_since_seen[person_id] += 1

        try:
            for keypoints in people:
                center = self.get_person_center(keypoints)
                if center is None:
                    continue
                
                person_id = self.assign_id(center)
                 # Extract keypoints
                right_shoulder = (int(keypoints[5][0]), int(keypoints[5][1]))
                right_elbow = (int(keypoints[7][0]), int(keypoints[7][1]))
                right_wrist = (int(keypoints[9][0]), int(keypoints[9][1]))
                left_shoulder = (int(keypoints[6][0]), int(keypoints[6][1]))
                left_elbow = (int(keypoints[8][0]), int(keypoints[8][1]))
                left_wrist = (int(keypoints[10][0]), int(keypoints[10][1]))
                right_hip = (int(keypoints[11][0]), int(keypoints[11][1]))
                left_hip = (int(keypoints[12][0]), int(keypoints[12][1]))
                
                # Calculate angles
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_elbow)
                left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_elbow)
                right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)
                left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)


                person_data = self.people_data[person_id]
                
                # Assuming person_data is a dictionary for each person tracking the reps and position
                # if right_elbow_angle is not None:  # Ensure the angle is defined
                    # Define the average values for right shoulder angle


                # Variable to track if we are in the "down" position
                in_down_position = False

                # Check the current right shoulder angle
                if right_shoulder_angle is not None:
                    # Transition from "up" (hands above) to "down" (hands below)
                    if right_shoulder_angle > 68.59 and person_data['position'] == 'up':
                        # Count a rep when hands go from above to below (down position)
                        if not in_down_position:  # Avoid counting multiple reps in a single cycle
                            person_data['position'] = 'down'
                            print(f"Rep {person_data['reps']} counted (hands down transition)")
                            person_data['reps'] += 1


                    # Transition from "down" (hands below) to "up" (hands above)
                    elif right_shoulder_angle < 68.59  and person_data['position'] == 'down'  :
                        person_data['position'] = 'up'


                # Store current right shoulder angle for next frame
                previous_right_shoulder_angle = right_shoulder_angle

                # Store current right shoulder angle for the next cycle
                previous_right_shoulder_angle = right_shoulder_angle

                
                self.log_frame_data(person_id, frame_number , right_shoulder_angle,left_shoulder_angle)
                
                cv2.putText(frame, f'ID: {person_id}', (right_shoulder[0] - 30, right_shoulder[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
                cv2.putText(frame, f'Reps: {person_data["reps"]}', 
                            (right_shoulder[0] - 30, right_shoulder[1] + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
                
                  # Show angles
                if right_elbow_angle is not None:
                    cv2.putText(frame, f'{int(right_elbow_angle)}', right_elbow, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_elbow_angle is not None:
                    cv2.putText(frame, f'{int(left_elbow_angle)}', left_elbow, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if right_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(right_shoulder_angle)}', right_shoulder, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_shoulder_angle is not None:
                    cv2.putText(frame, f'{int(left_shoulder_angle)}', left_shoulder, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if right_hip_angle is not None:
                    cv2.putText(frame, f'{int(right_hip_angle)}', right_hip, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if left_hip_angle is not None:
                    cv2.putText(frame, f'{int(left_hip_angle)}', left_hip, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw keypoints and connections
                cv2.circle(frame, right_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, right_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
                cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 2)
                cv2.circle(frame, right_hip, 5, (0, 255, 0), -1)
                cv2.line(frame, right_shoulder, right_hip, (0, 255, 0), 2)

                cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_elbow, 5, (255, 0, 0), -1)
                cv2.circle(frame, left_wrist, 5, (255, 0, 0), -1)
                cv2.line(frame, left_shoulder, left_elbow, (255, 0, 0), 2)
                cv2.line(frame, left_elbow, left_wrist, (255, 0, 0), 2)
                cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)
                cv2.line(frame, left_shoulder, left_hip, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing keypoints: {e}")
            
        self._cleanup_old_ids()
        return frame, None

    def _cleanup_old_ids(self):
        ids_to_remove = []
        for person_id in self.frames_since_seen:
            if self.frames_since_seen[person_id] > self.max_frames_missing:
                ids_to_remove.append(person_id)
        
        for person_id in ids_to_remove:
            self.frames_since_seen.pop(person_id)
            self.last_positions.pop(person_id)

    def log_frame_data(self, person_id, frame_number, right_shoulder_angle,left_shoulder_angle):
        current_time = time.time()
        time_elapsed = round(current_time - self.start_time, 2)
        self.frame_log.append([person_id, frame_number, time_elapsed, right_shoulder_angle,left_shoulder_angle])

    def save_csv(self, filename="jumping_jack_data.csv"):
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["person_id", "frame_number", "time_in_seconds", "right_shoulder_angle","left_shoulder_angle"])
            writer.writerows(self.frame_log)
        print(f"CSV saved to: {file_path}")

import argparse

def main(video_path, output_path):
    tracker = ExerciseTracker()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties (e.g., frame width, height, and FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the VideoWriter to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change this depending on your desired format (e.g., 'MJPG', 'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # frame = cv2.resize(frame, (1080, 720))
        processed_frame, message = tracker.process_frame(frame, frame_number)
        frame_number += 1
        
        # Display message if there's any
        if message:
            cv2.putText(processed_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame
        cv2.imshow('Exercise Tracker', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()  # Save the output video
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise Tracker")
    # Uncomment the next line to allow dynamic video path input
    # parser.add_argument('video_path', type=str, required=True, help="Path to the video file")
    # parser.add_argument('output_path', type=str, required=True, help="Path to save the output video")
    # args = parser.parse_args()
    
    # Here you can specify the video path and the output path manually
    main('/home/kranti/Documents/yoloV11_Exercise_Tracking/Input_Video/jumping_jack_3.mp4', '/home/kranti/Documents/yoloV11_Exercise_Tracking/Jumping_jack_multiplePerson_output1.mp4')
