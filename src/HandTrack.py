from socket import IP_MULTICAST_LOOP
import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import numpy as np
import copy
import itertools
import csv
import os
import time
from collections import deque
from model import KeyPointClassifier
import pyttsx3



class Track:

    def __init__(self, use_brect):
        self.next_frame = 0
        self.prev_frame = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.IMAGE_FILES = []
        self.use_brect = use_brect
        history_len = 16
        self.point_history = deque(maxlen=history_len)
        self.landmark_list = []
        self.num = 0
        self.mode = 0
        self.tts = ""
        self.frame_count = 0
        self.keypoint_classifier = KeyPointClassifier()
        with open(os.getcwd()+'\\src\\model\\keypoint_classifier\\keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

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
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                        brect = self.__calc_bounding_rect(debug_image, hand_landmarks)
                        # Normalize Joint Coordinates
                        self.landmark_list = self.__calc_landmark_list(image, hand_landmarks)
                        pre_processed_landmark_list = self.__pre_process_landmark(self.landmark_list)
                        pre_processed_point_history_list = self.__pre_process_point_history(debug_image, self.point_history)

                        # Dataset Generation
                        self.__makeCSV(self.num, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)

                        # Draw joints
                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                        debug_image = self.__draw_bounding_rect(self.use_brect, debug_image, brect)
                        debug_image = self.__draw_landmarks(debug_image, self.landmark_list)
                        debug_image = self.__draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id])
                        self.tts = self.__textBuilder(self.tts, key, self.keypoint_classifier_labels[hand_sign_id], self.frame_count)
                    
                else:
                    self.point_history.append([0, 0])

                self.frame_count = self.frame_count + 1
                self.next_frame = time.time()
                self.fps = 1/(self.next_frame-self.prev_frame)
                self.prev_frame = self.next_frame
                self.fps = int(self.fps)
                self.fps = str(self.fps)

                self.__ui(debug_image, str(self.fps), self.mode, self.num)
                if key == 46:
                    self.tts = ""

                text_size, _ = cv2.getTextSize("The current string is: " + self.tts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_w, text_h = text_size
                cv2.rectangle(debug_image, (0, 475 - text_h), (text_w, 480), (0,0,0), -1)
                cv2.putText(debug_image, "The current string is: " + self.tts, (0, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('Hand Tracking', debug_image)

        cap.release()
        cv2.destroyAllWindows()

    def __sel_mode(self, key, mode):
        num = -1
        if 97 <= key <= 122:  # a - z on keyboard Select class
            num = key - 97
        if key == 49:  # 1 - Normal operation
            mode = 0
        if key == 50:  # 2 - Train keypoint
            mode = 1
        if key == 51:  # 3
            mode = 2
        return num, mode

    def __calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

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

    def __makeCSV(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 26):
            csv_path = os.getcwd() + '\\src\\model\\keypoint_classifier\\keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 26):
            csv_path = os.getcwd() + '\\src\\model\\point_history_classifier\\point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def __draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangles
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

        return image

    def __draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 0, 0), 2)

            # Index finger
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 0, 0), 2)

            # Middle finger
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 0, 0), 2)

            # Ring finger
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 0, 0), 2)

            # Pinky
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 0, 0), 2)

            # Palm
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 0, 0), 2)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 0, 0), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 0), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def __draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]

        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def __ui(self, image, frames, mode, num):
        text = "FPS: {}  Resolution: {}x{}".format(frames, image.shape[1], image.shape[0])
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (636 - text_w, 0), (640, 4 + text_h), (0,0,0), -1)
        cv2.putText(image, text, (638 - text_w, 2 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        if mode == 0:
            text_size, _ = cv2.getTextSize('Press 2 to enter training mode, press ESC to exit', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
            cv2.putText(image, 'Press 2 to enter training mode, press ESC to exit', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        elif mode == 1:
            text_size, _ = cv2.getTextSize('Press 1 to exit training mode, press ESC to exit', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
            cv2.putText(image, 'Press 1 to exit training mode, press ESC to exit', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return

    def __textBuilder(self, tts, key, text, frame):

            #if key == 47: #Press '/' to add sign language input to string
                #tts = tts + text + ' ' #Adding a space for the text to speech to read individual letters
            
            if (frame%20) == 0:
                tts = tts + text + " "

            if key == 46: #Press '.' to clear string
                tts = ""

            if text == "v": #Read the current string and clear string
                engine = pyttsx3.init()
                engine.say(tts)
                engine.runAndWait()
                tts = ""

            return tts



