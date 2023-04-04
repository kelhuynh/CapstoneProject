"""
This module houses the Track class of which is used primarily for static gesture recognition
including video capture, input labeling and overlay, classification using keypoint and
point_history machine learning models.

MIT License

Copyright (c) 2023 Kelvin Huynh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__authors__ = ["Fred Zhu", "Jiahui Chen", "Kelvin Huynh", "Mirza Nafi Hasan", "Robert Zhu", "Zifan Meng"]
__date__ = "2023/04/04"
__deprecated__ = False
__license__ = "MIT License"
__status__ = "Prototype"
__version__ = "1.0"

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
from collections import Counter
from model import KeyPointClassifier
from model import PointHistoryClassifier
import pyttsx3


class Track:

    def __init__(self, use_brect):
        """
        Initialize Track object

        :param use_brect:
            Boolean that enables/disables bounding rectangle
        """

        # Instantiate Track object properties
        self.next_frame = 0
        self.prev_frame = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.IMAGE_FILES = []
        self.use_brect = use_brect
        self.history_len = 16
        self.point_history = deque(maxlen=self.history_len)
        self.finger_gesture_history = deque(maxlen=self.history_len)
        self.landmark_list = []
        self.num = 0
        self.mode = 0
        self.tts = ""
        self.frame_count = 0
        self.count = 0
        self.text = ""
        self.prev_key = 0

        # Instantiate Keypoint classification and PointHistory classification
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        with open(os.getcwd() + '\\model\\keypoint_classifier\\keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        with open(os.getcwd() + '\\model\\point_history_classifier\\point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [row[0] for row in self.point_history_classifier_labels]

    def motionTrack(self, cap, words: str):
        '''
        Initialize video capture

        :param cap:
            a cv2 VideoCapture object
        :param words:
            a string to be used for text to speech
        :returns:
            None
        '''
        self.tts = words
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while cap.isOpened():

                key = cv2.waitKey(10)
                if key == 49:
                    self.mode = 0
                    return self.tts, self.mode
                elif key == 50:
                    self.mode = 1
                    return self.tts, self.mode
                elif key == 51:
                    self.mode = 0
                elif key == 52:
                    self.mode = 1
                elif key == 27:
                    self.mode = 4
                    return self.tts, self.mode

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

                        # Hand sign classification
                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                        if self.mode == 0:  # Point gesture
                            self.point_history.append(self.landmark_list[8])
                        else:
                            self.point_history.append([0, 0])

                        # Finger gesture classification
                        self.finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (self.history_len * 2):
                            self.finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                        self.__makeCSV(self.num, self.mode, pre_processed_landmark_list, pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        self.finger_gesture_history.append(self.finger_gesture_id)
                        most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                        # Draw joints

                        debug_image = self.__draw_bounding_rect(self.use_brect, debug_image, brect)
                        debug_image = self.__draw_landmarks(debug_image, self.landmark_list)
                        if self.mode == 0:
                            self.tts = self.__textBuilder(self.tts, self.keypoint_classifier_labels[hand_sign_id], self.frame_count)
                            debug_image = self.__draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id], self.point_history_classifier_labels[most_common_fg_id[0][0]])

                else:
                    self.point_history.append([0, 0])

                self.next_frame = time.time()
                self.fps = 1 / (self.next_frame - self.prev_frame)
                self.prev_frame = self.next_frame
                self.frame_count = self.frame_count + 1  # Frame Counter
                self.fps = int(self.fps)
                self.fps = str(self.fps)

                self.__ui(debug_image, str(self.fps), self.mode, self.num)
                if key == 46:  # Press '.' to clear string
                    self.tts = ""

                if key == 8:  # Press 'Backspace' to clear last character
                    self.tts = self.tts[:-1]

                if key == 32:  # Press 'Space' to add a space
                    self.tts = self.tts + " "

                if (self.mode == 1):
                    text_size, _ = cv2.getTextSize("Added 0000 points for a", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_w, text_h = text_size
                    cv2.rectangle(debug_image, (635 - text_w, 475 - text_h), (640, 480), (0, 0, 0), -1)
                    if (97 <= key <= 122):
                        if self.prev_key != key:
                            self.count = 1
                            self.text = "Added {} points for {}".format(self.count, chr(key))
                        else:
                            self.count += 1
                            self.text = "Added {} points for {}".format(self.count, chr(key))
                        self.prev_key = key
                    cv2.putText(debug_image, self.text, (636 - text_w, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', debug_image)

    def __calc_bounding_rect(self, image, landmarks):
        '''
        Defines bounding rectangle for gesture

        :param image:
            an image frame read from a cv2 VideoCapture object
        :param landmarks:
            hand landmarks from data processed by a Mediapipe model
        :returns:
            an array of coordinates define the bounding rectangle around a gesture
        '''

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
        '''
        Shapes joint coordinate data for use in application

        :param image:
            an image frame read from a cv2 VideoCapture object
        :param landmarks:
            hand landmarks from data processed by a Mediapipe model
        :returns:
            an array with coordinate data in desired format for processing
        '''

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
        '''
        Makes a copy of an array of coordinates corresponding to hand joints and normalizes the coordinates

        :param landmark_list:
            an array returned from __calc_landmark_list
        :returns:
            an array of joint coordinate data that has been normalized
        '''

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
            '''
            Normalizes coordinate points against the max value in the array

            :param n:
                a float value corresponding to one hand joint coordinate
            :returns:
                a normalized hand joint coordinate
            '''
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def __pre_process_point_history(self, img, point_history):
        '''
        Makes a copy of a 2D-array of coordinates corresponding to prior and currently detected hand joints and normalizes the coordinates

        :param img:
            an image frame read from a cv2 VideoCapture object
        :param landmark_list:
            an array containing past hand joint coordinates
        :returns:
            an array of joint coordinate data that has been normalized
        '''

        img_width, img_height = img.shape[1], img.shape[0]
        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / img_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / img_height

        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def __makeCSV(self, number, mode, landmark_list, point_history_list):
        '''
        Exports normalized coordinate data into corresponding csv file depending on mode

        :param number:
            an integer corresponding to the label classification of the data to be exported
        :param mode:
            an integer that details the current mode of operation
        :param landmark_list:
            an array of normalized keypoint data for static detection
        :param point_history_list:
            an array of normalized hand joint data for motion detection
        :returns:
            None
        '''

        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 36):
            csv_path = os.getcwd() + '\\model\\keypoint_classifier\\keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 26):
            csv_path = os.getcwd() + '\\model\\point_history_classifier\\point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def __draw_bounding_rect(self, use_brect, image, brect):
        '''
        Overlay for bounding rectangle

        :param use_brect:
            Boolean that determines whether a bounding rectangle should be drawn or not
        :param image:
            an image frame read from a cv2 VideoCapture object
        :param brect:
            an array containing the coordinates of the bounding box
        :returns:
            an image frame that has been overlaid with the bounding rectangle
        '''

        if use_brect:
            # Outer rectangles / bounding box
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

        return image

    def __draw_landmarks(self, image, landmark_point):
        '''
        Customized hand joint overlay

        :param image:
            an image frame read from a cv2 VideoCapture object
        :param landmark_point:
            an array of joint coordinate data
        :returns:
            an image that has been overlaid with connecting lines and circles depicting hand joint locations
        '''
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

    def __draw_info_text(self, image, brect, handedness, hand_sign_text, hand_gesture_text):
        '''
        Draws hand gesture classification onto the image frame

        :param image:
            an image frame read from a cv2 VideoCapture object
        :param brect:
            an array containing coordinate data for a bounding box
        :param handedness:
            a property of processed data from Mediapipe hands model. This determines whether the left or right hand was used by the user to sign
        :param hand_sign_text:
            a string that identifies the static gesture that was captured
        :param hand_gesture_text:
            a string that identifies the motion gesture that was captured
        :returns:
            an image that has been overlaid with information about the captured gesture
        '''

        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]

        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def __ui(self, image, frames: str, mode: int, num: int):
        '''
        Defines the user interface to be shown when capturing video input

        :param image:
            an image frame read from a cv2 VideoCapture object
        :param frames:
            a string that contains the number of frames read from a VideoCapture object
        :param mode:
            an integer that defines the current operation mode of the translator
        :param num:
        :returns:
            None
        '''
        text = "FPS: {}  Resolution: {}x{}".format(frames, image.shape[1], image.shape[0])
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (636 - text_w, 0), (640, 4 + text_h), (0, 0, 0), -1)
        cv2.putText(image, text, (638 - text_w, 2 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if mode == 0:
            text_size, _ = cv2.getTextSize('Press 1 for Dynamic, 4 for Training, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
            cv2.putText(image, 'Press 1 for Dynamic, 4 for Training, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            text_size, _ = cv2.getTextSize("The current string is: " + self.tts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0, 475 - text_h), (text_w, 480), (0, 0, 0), -1)
            cv2.putText(image, "The current string is: " + self.tts, (0, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif mode == 1:
            text_size, _ = cv2.getTextSize('Press 1 for Translation, 3 to Point History, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
            cv2.putText(image, 'Press 1 for Translation, 3 to Static, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif mode == 2:
            text_size, _ = cv2.getTextSize('Press 1 for Translation, 2 to Training Mode, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
            cv2.putText(image, 'Press 1 for Translation, 2 to Static, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return

    def __textBuilder(self, tts: str, text: str, frame):
        '''
        Builds the text to speech string

        :param tts:
            the string to be appended to
        :param text:
            the text to be appended to tts
        :param frame:
            an integer containing the number of frames counted
        :returns:
            a new text to speech string
        '''

        if (frame % 40) == 0:  # Modify this value for string record frequency
            tts = tts + text + " "

        return tts
