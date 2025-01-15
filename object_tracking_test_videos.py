import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time

# Initialize Object Detection
od = ObjectDetection()

# Función para analizar un video durante 5 segundos y contar vehículos
def analyze_video(video_path, target_ids):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_to_run = int(3 * video_fps)

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

        # Calcular FPS
        current_time = time.time()
        frame_id += 1

        if (current_time - fps_start_time) > 1:
            fps = frame_id
            frame_id = 0
            fps_start_time = current_time

        # Mostrar los FPS en el frame
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        center_points_cur_frame = []

        # Detectar objetos en el frame
        (class_ids, scores, boxes) = od.detect(frame)
        for i, box in enumerate(boxes):
            if class_ids[i] not in target_ids:
                continue  # Saltar si la id no es 2, 5,

            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Comparar frames al principio
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

        # Hacer una copia de los puntos
        center_points_prev_frame = center_points_cur_frame.copy()

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return max(tracking_objects.keys(), default=-1) + 1

# IDs que deseas rastrear
target_ids = {2, 5, 7}

# Analizar el primer video y obtener el número de vehículos detectados
#vehicles_in_first_video = analyze_video("1.mp4", target_ids)


# Analizar el segundo video y obtener el número de vehículos detectados
#vehicles_in_second_video = analyze_video("2.mp4", target_ids)


# Analizar el primer video y obtener el número de vehículos detectados
#vehicles_in_third_video = analyze_video("3.mp4", target_ids)


# Analizar el segundo video y obtener el número de vehículos detectados
#vehicles_in_fourth_video = analyze_video("4.mp4", target_ids)


# Analizar el primer video y obtener el número de vehículos detectados
#vehicles_in_fifth_video = analyze_video("5.mp4", target_ids)


# Analizar el segundo video y obtener el número de vehículos detectados
#vehicles_in_sixth_video = analyze_video("6.mp4", target_ids)


# Analizar el primer video y obtener el número de vehículos detectados
#vehicles_in_seventh_video = analyze_video("7.mp4", target_ids)


# Analizar el segundo video y obtener el número de vehículos detectados
#vehicles_in_eighth_video = analyze_video("8.mp4", target_ids)


# Analizar el primer video y obtener el número de vehículos detectados
# vehicles_in_ninth_video = analyze_video("9.mp4", target_ids)


# Analizar el segundo video y obtener el número de vehículos detectados
#vehicles_in_tenth_video = analyze_video("10.mp4", target_ids)


# print(f"Vehículos detectados en el primer video: {vehicles_in_first_video}")
# print(f"Vehículos detectados en el segundo video: {vehicles_in_second_video}")
# print(f"Vehículos detectados en el tercer video: {vehicles_in_third_video}")
# print(f"Vehículos detectados en el cuarto video: {vehicles_in_fourth_video}")
# print(f"Vehículos detectados en el quinto video: {vehicles_in_fifth_video}")
# print(f"Vehículos detectados en el sexto video: {vehicles_in_sixth_video}")
# print(f"Vehículos detectados en el septimo video: {vehicles_in_seventh_video}")
# print(f"Vehículos detectados en el octavo video: {vehicles_in_eighth_video}")
# print(f"Vehículos detectados en el noveno video: {vehicles_in_ninth_video}")
# print(f"Vehículos detectados en el decimo video: {vehicles_in_tenth_video}")