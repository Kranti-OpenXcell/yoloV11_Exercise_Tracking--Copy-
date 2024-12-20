import os
import cv2
import time
import numpy as np
import argparse
from pathlib import Path
from random import randint
import torch.backends.cudnn as cudnn
from ultralytics import YOLO

from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path

# For SORT tracking
from sort import *

#............................... Bounding Boxes Drawing ............................
def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        
        if save_with_object_id:
            txt_str = "%i %i %f %f %f %f %f %f\n" % (
                id, cat, x1/img.shape[1], y1/img.shape[0], x2/img.shape[1], y2/img.shape[0],
                (x1 + (x2-x1)/2)/img.shape[1], (y1 + (y2-y1)/2)/img.shape[0])
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

def detect(opt):
    source = opt.source
    save_img = not opt.nosave
    save_txt = opt.save_txt
    save_with_object_id = opt.save_with_object_id
    
    # Initialize SORT
    sort_tracker = Sort(max_age=opt.sort_max_age,
                       min_hits=opt.sort_min_hits,
                       iou_threshold=opt.sort_iou_thresh)
    
    # Generate random colors for tracking visualization
    rand_color_list = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(5003)]
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(opt.weights)
    names = model.names
    
    # Video setup
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    if webcam:
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source}")
            return
            
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer setup
    if save_img:
        vid_writer = cv2.VideoWriter(
            str(save_dir / f'tracked_{Path(source).stem}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
        )
    
    t0 = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete")
            break
            
        frame_count += 1
        
        # YOLOv8 inference
        results = model.predict(frame, conf=opt.conf_thres, iou=opt.iou_thres, classes=opt.classes)[0]
        
        # Process detections
        if len(results.boxes) > 0:
            # Get boxes, confidence scores, and class ids
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            # Prepare detections for SORT
            dets_to_sort = np.hstack((boxes, confidences[:, np.newaxis], class_ids[:, np.newaxis]))
            
            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)
            
            # Draw boxes and tracks
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                
                # Draw boxes
                frame = draw_boxes(frame, bbox_xyxy, identities, categories, names, 
                                 save_with_object_id, str(save_dir / 'labels' / f'frame_{frame_count}'))
                
                # Draw tracks if enabled
                if opt.colored_trk:
                    for track in sort_tracker.getTrackers():
                        for i in range(len(track.centroidarr)-1):
                            cv2.line(frame, 
                                   (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                                   (int(track.centroidarr[i+1][0]), int(track.centroidarr[i+1][1])),
                                   rand_color_list[track.id % len(rand_color_list)], 2)
        
        # Display results
        if True:
                target_width, target_height = 1280, 720

                # Resize the frame to the target dimensions
                im0_resized = cv2.resize(frame, (target_width, target_height))

                cv2.imshow("Video Output", im0_resized)  # Use a consistent window name
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Exiting video display...")
                    break  # Gracefully exit the loop
        
        # Save results
        if save_img:
            vid_writer.write(frame)
    
    # Cleanup
    if save_img:
        vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11n.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--sort-max-age', type=int, default=5, help='SORT maximum age')
    parser.add_argument('--sort-min-hits', type=int, default=2, help='SORT minimum hits')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2, help='SORT IOU threshold')
    
    opt = parser.parse_args()
    print(opt)
    
    detect(opt)