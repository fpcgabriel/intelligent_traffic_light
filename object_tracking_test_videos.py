import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time

# Initialize Object Detection
od = ObjectDetection()

def analyze_video(video_path, target_ids, secs = 5):
    ''' Function to get the number of objects found in a video
    inputs: 
    - the computer path of the video (string)
    - the ids of classes searched (list)
    - secs: number of seconds of the video wanted to analyze (int)
    '''
    cap = cv2.VideoCapture(video_path) # Captures the video
    video_fps = cap.get(cv2.CAP_PROP_FPS) # Returns the fps of the video
    num_frames_to_run = int(secs * video_fps) # Computes the number of frames that will be analyzed
    #Inicialization of variables
    count = 0
    center_points_prev_frame = []
    tracking_objects = {}
    track_id = 0
    fps_start_time = 0
    fps = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret or count > num_frames_to_run:
            break

        # Computing FPS
        current_time = time.time()
        frame_id += 1

        if (current_time - fps_start_time) > 1:
            fps = frame_id
            frame_id = 0
            fps_start_time = current_time

        # Showing FPS in frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        center_points_cur_frame = []

        # Detects objects in frames
        (class_ids, scores, boxes) = od.detect(frame)
        for i, box in enumerate(boxes):
            if class_ids[i] not in target_ids:
                continue  # Ignores other classes rather than selected
            # Drawing boxes with objects detected
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Gets the center point of the object
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))

        # Comparing frames
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                if not object_exists:
                    tracking_objects.pop(object_id)

            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        print("Tracking objects")
        print(tracking_objects)
        print("CUR FRAME LEFT PTS")
        print(center_points_cur_frame)
        print("FPS del video:", fps)
        
        cv2.imshow("Frame", frame)

        # Current frame to previous frame
        center_points_prev_frame = center_points_cur_frame.copy()

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return max(tracking_objects.keys(), default=-1) + 1

# Classes will be tracked (cars, buses and motorcycles)
target_ids = {2, 5, 7}
vehicles_in_first_video = analyze_video("los_angeles.mp4", target_ids)