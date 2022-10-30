import cv2
import mediapipe as mp


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
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

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
                for hand_landmarks in results.multi_hand_landmarks:
                    print("Landmarks:", hand_landmarks)
                    print(
                        'Index finger tip coordinates: (',
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
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
