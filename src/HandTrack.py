import cv2
import mediapipe as mp
import copy
import itertools
import csv
from collections import deque


class Track:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.IMAGE_FILES = []
        history_len = 16
        self.point_history = deque(maxlen=history_len)
        self.landmark_list = []
        self.num = 0
        self.mode = 0

    def motionTrack(self):
        cap = cv2.VideoCapture(0)
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while cap.isOpened():

                key = cv2.waitKey(10)
                if key == 27:
                    break
                self.num, self.mode = self.__sel_mode(key, self.mode)

                success, image = cap.read()  # Success = feed established
                if not success:
                    print("Empty camera frame")
                    continue

                # Performance Optimization
                image = cv2.flip(image, 1)
                debug_image = copy.deepcopy(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image.flags.writeable = False

                results = hands.process(image)

                # Draw Hand joints

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multihandedness):

                        # Normalize Joint Coordinates
                        self.landmark_list = self.__calc_landmark_list(image, hand_landmarks)
                        pre_processed_landmark_list = self.__pre_process_landmark(self.landmark_list)
                        pre_processed_point_history_list = self.__pre_process_point_history(debug_image, self.point_history)

                        # Dataset Generation
                        self.__makeCSV(self.num, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)

                cv2.imshow('Hand Tracking', debug_image)

        cap.release()

    def __sel_mode(key, mode):
        num = -1
        if 48 <= key <= 57:  # 0 ~ 9
            num = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return num, mode

    def __calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def __pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def __pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def __makeCSV(number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return
