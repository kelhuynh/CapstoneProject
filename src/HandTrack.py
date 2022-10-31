import cv2
import mediapipe as mp
import os
import numpy as np


class Track:

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.IMAGE_FILES = []

    def motionTrack(self):
        cap = cv2.VideoCapture(0)
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, image = cap.read()  # Success = feed established
                if not success:
                    print("Empty camera frame")
                    continue

                # Performance Optimization

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw Hand joints

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        print(f'HAND NO.: {hand_no+1}')
                        print("----------------------")
                        for point in self.mp_hands.HandLandmark:  # Gives normalized coordinates for hand landmark
                            normalized_landmark = hand_landmarks.landmark[point]
                            print(point)
                            print(normalized_landmark)

                cv2.imshow('Hand Tracking', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:  # Continuous feed until esc is pressed
                    break

        cap.release()

    def imageDraw(self, image_files):  # To be used for dataset creation
        self.IMAGE_FILES = image_files
        with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:

            for index, file in enumerate(self.IMAGE_FILES):
                image = cv2.flip(cv2.imread(file), 1)  # Flip image to match correct handedness
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                print("Handedness:", results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):  # TODO export normalized coordinates to .csv for dataset
                    print(f'HAND NO.: {hand_no+1}')
                    print("----------------------")
                    for point in self.mp_hands.HandLandmark:
                        normalized_landmark = hand_landmarks.landmark[point]
                        print(normalized_landmark)

                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                cv2.imwrite('/tmp/annotated_image' + str(index) + '.png', cv2.flip(annotated_image, 1))

                if not results.multi_hand_world_landmarks:
                    continue
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    self.mp_drawing.plot_landmarks(hand_world_landmarks, self.mp_hands.HAND_CONNECTIONS, azimuth=5)

    def imageProcess(self, path):
        img = cv2.imread(path)
        img_To_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_Flip = cv2.flip(img_To_RGB, 1)

        with self.mp_hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:

            output = hands.process(img_Flip)

            hands.close()

        try:
            data = output.multi_hand_landmarks[0]
            data = str(data)
            data = data.strip().split('\n')

            garbage = ['Landmark {', ' visibility: 0.0', ' presence: 0.0', '}']
            no_garbage = []

            for i in data:
                if i not in garbage:
                    no_garbage.append(i)

            clean = []

            for i in no_garbage:
                i = i.strip()
                clean.append(i[2:])

            for i in range(0, len(clean)):
                clean[i] = float(clean[i])

            return ([clean])

        except Exception as ex:
            print("Exception in imageProcess: ", ex)
            return (np.zeros([1, 63], dtype=int)[0])

    def makeCSV(self, path):
        file_name = open('dataset.csv', 'a')

        for folder in os.listdir(path):
            if '._' in folder:
                pass
            else:
                for number in os.listdir(path + '/' + folder):
                    if '._' in number:
                        pass
                    else:
                        label = folder
                        file_location = path + '/' + folder + '/' + number

                        data = self.imageProcess(file_location)

                        try:
                            for i in data:
                                file_name.write(str(i))
                                file_name.write(',')

                            file_name.write(label)
                            file_name.write('\n')

                        except Exception as ex:
                            print("Exception in makeCSV: ", ex)

                            file_name.write('0')
                            file_name.write(',')

                            file_name.write('None')
                            file_name.write('\n')

        file_name.close()
        print('Data added to csv')
