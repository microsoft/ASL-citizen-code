import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import csv
from timeit import default_timer as timer
import os

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#Update paths here
src_path = '../data//videos/'
dst_path = '../data/pose_files/'
data_csv  = '../data_csv/aslcitizen_training_set.csv'

with open(data_csv, 'r') as file:
    reader = csv.reader(file)
    count_f = 0
    start = timer()
    for row in reader:
        f = src_path + row[2]

        with mp_holistic.Holistic(
            static_image_mode=False, min_detection_confidence=0.5) as holistic:
            
            video = cv2.VideoCapture(f)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

            feature = np.zeros((int(total_frames), 543, 2))
            count = 0
            success = 1

            while success: 
                success, image = video.read()
                if success:
                    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    for i in range(33):
                       if results.pose_landmarks:
                            feature[count][i][0] = results.pose_landmarks.landmark[i].x
                            feature[count][i][1] = results.pose_landmarks.landmark[i].y

                    j = 33
                    for i in range(21):
                        if results.right_hand_landmarks:
                           feature[count][i+j][0] = results.right_hand_landmarks.landmark[i].x
                           feature[count][i+j][1] = results.right_hand_landmarks.landmark[i].y

                    j = 54
                    for i in range(21):
                        if results.left_hand_landmarks:
                            feature[count][i+j][0] = results.left_hand_landmarks.landmark[i].x
                            feature[count][i+j][1] = results.left_hand_landmarks.landmark[i].y

                    j = 75
                    for i in range(468):
                        if results.face_landmarks:
                            feature[count][i+j][0] = results.face_landmarks.landmark[i].x
                            feature[count][i+j][1] = results.face_landmarks.landmark[i].y
                    count += 1

            name = ''.join(row[2][:-4].split('/'))
            np.save(dst_path + name + '.npy', feature)            
            count_f += 1
            if count_f % 10 == 0:
                end = timer()
                print(end - start)
                print(count_f)
                start = end