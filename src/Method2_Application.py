import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pyttsx3
from collections import deque
import tensorflow as tf

mp_holistic = mp.solutions.holistic  # Instatiate Mediapiep Holistic model
mp_drawing = mp.solutions.drawing_utils  # Instantiate Mediapipe Drawing utilities
model = tf.keras.models.load_model('action.h5')

# Initialize Global Variables

next_frame = 0
prev_frame = 0
history_len = 16
num = 0
mode = 0
frame_count = 0
count = 0
prev_key = 0
spot = 0
tts = ""  # Blank string used for TTS
text = ""  # String used for classification
finger_gesture_history = deque(maxlen=history_len)
short = True


def mediapipe_detection(image, model):
    '''
    Processes an image frame through a Mediapipe model

    :param image:
        an image frame read from a cv2 VideoCapture object
    :param model:
        a Mediapipe model that processes the image frame
    :returns:
        - image - a processed image frame
        - results - processed data from a Mediapipe model
    '''

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    '''
    Body and hand joint recognition overlay

    :param image:
        an image frame read from a cv2 VideoCapture object
    :param results:
        processed data from a Mediapipe model
    :returns:
        None
    '''

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    '''
    Customized body and hand joint recognition overlay

    :param image:
        an image frame read from a cv2 VideoCapture object
    :param results:
        processed data from a Mediapipe model
    :returns:
        None
    '''

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # Draw right hand connections

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    '''
    Combines keypoint data from Mediapipe into a single array

    :param results:
        processed data from a Mediapipe model
    :returns:
        an array containing coordinates correlating to the user's full body, left hand, and right hand
    '''
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


def sel_mode(key, mode):
    """
    Determines the operation mode of the translator

    :param key:
        an integer that stores the ordinal value of a keypress. For example, the letter a would be assigned a value of 97
    :param mode:
        an integer that stores the current operation mode of the translator
    :returns:
        - num - an integer corresponding to a label classification based on the keypress
        - mode - an integer that determines the current mode of operation of the translator
    """

    num = -1
    if key == 49:  # 1 - Normal operation
        mode = 0
    if key == 50:  # 2 - Train keypoint
        mode = 1
    return num, mode


def ui(image, frames: str, mode: int, text_string: str):
    '''
    Defines the user interface to be shown when capturing video input

    :param image:
        an image frame read from a cv2 VideoCapture object
    :param frames:
        a string that contains the number of frames read from a VideoCapture object
    :param mode:
        an integer that defines the current operation mode of the translator
    :param text_string:
        the string to be displayed on the user interface. This is the text to speech string
    :returns:
        None
    '''

    text = "FPS: {}  Resolution: {}x{}".format(frames, image.shape[1], image.shape[0])
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_w, text_h = text_size
    cv2.rectangle(image, (636 - text_w, 0), (640, 4 + text_h), (0, 0, 0), -1)
    cv2.putText(image, text, (638 - text_w, 2 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    if mode == 0:
        global spot, short
        multiplier = 1
        line1 = ''
        line2 = ''
        text_size, _ = cv2.getTextSize('Press 2 to Training Mode ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
        cv2.putText(image, 'Press 2 to Training Mode, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        text_size, _ = cv2.getTextSize("The current string is: " + text_string, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        if (text_w > 640):
            multiplier = 2
            if spot == 0:
                spot = len(text_string.split()) - 1
            for i in range(0, spot):
                line1 = line1 + text_string.split()[i] + ' '
            for i in range(spot, len(text_string.split())):
                line2 = line2 + text_string.split()[i] + ' '
            text2_size, _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text2_w, text2_h = text2_size
            if ((640 - text2_w) < 10 or text2_w > 640):
                short = False
        else:
            line1 = text_string
        cv2.rectangle(image, (0, 480 - (text_h + 5) * multiplier), (text_w, 480), (0, 0, 0), -1)
        cv2.putText(image, "The current string is: " + line1, (0, 476 - 12 * (multiplier - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, line2, (0, 476 - 12 * (multiplier - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    elif mode == 1:
        text_size, _ = cv2.getTextSize('Press 1 for Translation, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
        cv2.putText(image, 'Press 1 for Translation, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return


def textBuilder(tts: str, text: str, frame):
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

    if (frame % 20) == 0:  # If the number of frames counted is a multiple of 20 then build the text to speech string
        if len(tts.split()) == 0 or text != tts.split()[-1]:  # Check to see if last classified word is unique from new classification
            if text == "noinput":  # If the predicted text is 'noinput' then do not append anything to the tts string
                pass
            else:
                tts = tts + text + " "
    return tts


def textToSpeech(tts: str, text: str):
    '''
    Runs the text to speech engine if the last predicted text is "speak"

    :param tts:
        the string to be read by the text to speech engine
    :param text:
        a string that is used to trigger the text to speech engine
    :returns:
        a cleared text to speech string
    '''

    if text == "speak":  # Read the current string and clear string
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.say(tts)
        engine.runAndWait()
        engine.stop()
        tts = ""
        global spot, short
        spot = 0
        short = True

    return tts


def UserInput(image):
    '''
    Capture input from user keypresses and store into a string

    :param image:
        an image frame that is read from a cv2 VideoCapture object
    :returns:
        the text string that is input by the user during video capture
    '''

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = ""  # Set the initial text to an empty string
    while True:
        image = np.zeros((480, 640, 3), dtype=np.uint8)  # Remove previous text from the image

        # Display the current text on the screen
        cv2.putText(image,
                    "Enter an available word or new word to train: " + text,
                    org=(0, 476 - 12),
                    fontFace=font,
                    fontScale=0.4,
                    color=(0, 0, 255),
                    thickness=1)
        text_size, _ = cv2.getTextSize('Press 1 for Translation, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
        cv2.putText(image, 'Press 1 for Translation, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Wait for a key press
        key = cv2.waitKey(1)

        # If the key pressed is a letter or a space, add it to the text
        if (key == 32) or (key >= 65 and key <= 122):
            text += chr(key)

        # If the key pressed is backspace, delete the last character
        elif key == 8:
            text = text[:-1]

        # If the key pressed is enter, exit the loop and return the text
        elif key == 13:
            break

        # elif key == 49:
            # mode = 0
        cv2.imshow('OpenCV Feed', image)

    return text


def countdown_timer(cap, countdown_time):
    '''
    Display countdown timer on Video Capture

    :param cap:
        CV2 VideoCapture object
    :param countdown_time:
        an integer that correlates to the amount of time to be waited
    :returns:
        None
    '''
    # Set the font and text size for the countdown timer
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 255)

    # Read initial image frame from VideoCapture object
    ret, frame = cap.read()

    # Display initial image
    cv2.imshow('OpenCV Feed', frame)

    # Loop for countdown display
    for i in range(countdown_time, 0, -1):

        # Read new frame
        ret, frame = cap.read()

        # Clear the current display image and display the current time remaining
        text = 'Starting in {}...'.format(i)
        text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, 2)

        # Display image and wait for 1 second
        cv2.imshow('OpenCV Feed', frame)
        cv2.waitKey(1000)


# Define file path to be used for Numpy Arrays
cwd = os.getcwd()
DATA_PATH = os.path.join('MP_Data')


actions = np.loadtxt("actions.txt", dtype='str')  # Load action array used for classification
no_sequences = 30  # Number of video captures to be completed
sequence_length = 30  # Number of frames per video
sequence = []  # Initialize list to store 30 frames worth of data
sentence = ['']  # Initialize list to store gesture classifcation
predictions = []  # Initialize list to store predicted gestures
threshold = 0.5  # Confidence threshold to be used when predicting gesture
exit_flag = False  # Boolean used as a checking condition for exiting function scopes

cap = cv2.VideoCapture(0)  # Initialize VideoCapture object

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        key = cv2.waitKey(10)
        if key == 27:
            break
        num, mode = sel_mode(key, mode)

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        if mode == 0:  # Testing mode
            exit_flag = False
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

            # 3. Gesture classification logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            next_frame = time.time()
            fps = 1 / (next_frame - prev_frame)
            prev_frame = next_frame
            frame_count = frame_count + 1  # Frame Counter
            fps = int(fps)
            fps = str(fps)

            if short:
                tts = textBuilder(tts, sentence[-1], frame_count)
            tts = textToSpeech(tts, sentence[-1])
            ui(image, str(fps), mode, tts)

            if key == 46:  # Press '.' to clear text to speech string
                tts = ""

            if key == 8:  # Press 'Backspace' to clear last character in the text to speech string
                tts = tts[:-1]

            if key == 32:  # Press 'Spacebar' to add a space to the text to speech string
                tts = tts + " "

            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)  # Set capture window to always be on top

        elif mode == 1:
            start_folder = 0
            action = UserInput(image)
            if action in actions:
                dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
                for sequence in range(1, no_sequences + 1):
                    try:
                        os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
                    except (FileExistsError, OSError):
                        pass
            else:  # New Action -> First time runthrough

                actions = np.append(actions, action)
                np.savetxt('actions.txt', actions, fmt='%s')
                os.mkdir(os.path.join(DATA_PATH, action))

                dirmax = 0
                for sequence in range(0, no_sequences + 1):
                    try:
                        os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
                    except (FileExistsError, OSError):
                        pass

            # Declare countdown timer for training
            countdown_timer(cap, 3)

            for sequence in range(dirmax, dirmax + no_sequences + 1):
                # Loop through video length aka sequence length
                if exit_flag:
                    break
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    text_size, _ = cv2.getTextSize('Press 1 for Translation', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_w, text_h = text_size
                    cv2.rectangle(image, (0, 0), (0 + text_w, 2 + text_h), (0, 0, 0), -1)
                    cv2.putText(image, 'Press 1 for Translation', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    # Wait in-between each frame
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 476 - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 476 - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break out of training
                    if cv2.waitKey(10) & 0xFF == ord('1'):
                        exit_flag = True
                        break

            # Reset mode, frames collected, predicted word, and predictions
            mode = 0
            sequence = []
            sentence = ['']
            predictions = []

    cv2.imshow('OpenCV Feed', image)
    cap.release()
    cv2.destroyAllWindows()
