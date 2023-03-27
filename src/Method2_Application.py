import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pyttsx3
from collections import deque
import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from tensorflow.python.keras.utils import to_categorical
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import LSTM, Dense
#from tensorflow.python.keras.callbacks import TensorBoard


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
model = tf.keras.models.load_model('action.h5')

next_frame = 0
prev_frame = 0
tts = ""
history_len = 16
num = 0
mode = 0
frame_count = 0
count = 0
text = ""
prev_key = 0
finger_gesture_history = deque(maxlen=history_len)
global spot
spot = 0
global tooLong
short = True


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def sel_mode(key, mode):
    num = -1
    if key == 49:  # 1 - Normal operation
          mode = 0
    if key == 50:  # 2 - Train keypoint
         mode = 1
    return num, mode

def ui(image, frames, mode, text_string):
    text = "FPS: {}  Resolution: {}x{}".format(frames, image.shape[1], image.shape[0])
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_w, text_h = text_size
    cv2.rectangle(image, (636 - text_w, 0), (640, 4 + text_h), (0,0,0), -1)
    cv2.putText(image, text, (638 - text_w, 2 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    if mode == 0:
        global spot
        multiplier = 1
        line1 = ''
        line2 = ''
        text_size, _ = cv2.getTextSize('Press 2 to Training Mode ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
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
            if ((640 - text2_w) < 10 or text2_w > 640): short = False
        else: line1 = text_string
        cv2.rectangle(image, (0, 480 - (text_h + 5) * multiplier), (text_w, 480), (0,0,0), -1)
        cv2.putText(image, "The current string is: " + line1, (0, 476 - 12 * (multiplier - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, line2, (0, 476 - 12 * (multiplier - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    elif mode == 1:
        text_size, _ = cv2.getTextSize('Press 1 for Translation, ESC to Exit Program', cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
        cv2.putText(image, 'Press 1 for Translation, ESC to Exit Program', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return

def textBuilder(tts, text, frame):
    if (frame%20) == 0: #Modify this value for string record frequency
       if len(tts.split()) == 0 or text != tts.split()[-1]: # Check if 2last word is different from new word
            if text == "noinput":
                pass
            else:
                tts = tts + text + " "
                """ engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop() """
    return tts
    
def textToSpeech(tts, text):

    if text == "speak": #Read the current string and clear string
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        voices = engine.getProperty('voices') 
        engine.setProperty('voice', voices[1].id)
        engine.say(tts)
        engine.runAndWait()
        engine.stop()
        tts = ""
        global spot
        spot = 0
        global short
        short = True

    return tts

def UserInput(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
# Set the initial text to an empty string
    text = ""
    while True:
    # Remove previous text from the image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
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
        cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
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

        #elif key == 49:
            #mode = 0
        cv2.imshow('OpenCV Feed', image)
        
    return text

def countdown_timer(cap, countdown_time):
    # Set the font and text size for the countdown timer
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 255)

    # Capture initial frame
    ret, frame = cap.read()
    
    # Display initial image
    cv2.imshow('OpenCV Feed', frame)

    # Loop through the countdown timer
    for i in range(countdown_time, 0, -1):
        # Capture new frame
        ret, frame = cap.read()

        # Clear the display image and display the current countdown time
        text = 'Starting in {}...'.format(i)
        text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, 2)

        # Display the image and wait for 1 second
        cv2.imshow('OpenCV Feed', frame)
        cv2.waitKey(1000)
    
# Path for exported data, numpy arrays
cwd = os.getcwd()
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.loadtxt("actions.txt", dtype='str')

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

sequence = []
sentence = ['']
predictions = []
threshold = 0.5
exit_flag = False

""" colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame """

cap = cv2.VideoCapture(0)
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

        if mode == 0: #Testing mode
            exit_flag = False
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                #image = prob_viz(res, actions, image, colors)
        
            next_frame = time.time()
            fps = 1/(next_frame-prev_frame)
            prev_frame = next_frame
            frame_count = frame_count + 1 #Frame Counter
            fps = int(fps)
            fps = str(fps)
            
            if short:
                tts = textBuilder(tts, sentence[-1], frame_count)
            tts = textToSpeech(tts, sentence[-1]) 
            ui(image, str(fps), mode, tts)

            if key == 46: #Press '.' to clear string
                tts = ""
                
            if key == 8: #Press 'Backspace' to clear last character
                tts = tts[:-1]
                
            if key == 32: #Press 'Space
                tts = tts + " "
             
            #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            #cv2.putText(image, ' '.join(sentence), (3,30), 
            #cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)

        elif mode == 1:
            start_folder = 0
            action = UserInput(image)
            if action in actions:
                dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
                for sequence in range(1,no_sequences+1):
                    try: 
                        os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
                    except:
                        pass
            else:  # New Action -> First time runthrough

                actions = np.append(actions, action)
                np.savetxt('actions.txt', actions, fmt='%s')  
                os.mkdir(os.path.join(DATA_PATH, action))

                dirmax = 0
                for sequence in range(0,no_sequences+1):
                    try: 
                        os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
                    except:
                        pass
            
            #Timer
            countdown_timer(cap, 3)

            for sequence in range(dirmax, dirmax+no_sequences+1):
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
                    cv2.rectangle(image, (0,0), (0 + text_w, 2 + text_h), (0,0,0), -1)
                    cv2.putText(image, 'Press 1 for Translation', (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                

                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
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
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('1'):
                        exit_flag = True
                        break
            
            mode = 0
            sequence = []
            sentence = ['']
            predictions = []

        '''if (mode == 1):
            text_size, _ = cv2.getTextSize("Added 0000 points for a", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_w, text_h = text_size
            cv2.rectangle(image, (635 - text_w, 475 - text_h), (640, 480), (0,0,0), -1)
            if (97 <= key <= 122):
                if prev_key != key:
                    count = 1
                    text = "Added {} points for {}".format(count,chr(key))                     
                else:
                    count += 1
                    text = "Added {} points for {}".format(count,chr(key))
                prev_key = key
            cv2.putText(image, text, (636 - text_w, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)'''

    cv2.imshow('OpenCV Feed', image)              
    cap.release()
    cv2.destroyAllWindows()