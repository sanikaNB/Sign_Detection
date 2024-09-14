import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh]))
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

actions = np.array(['A','B','C','D','E','F','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
# 'A','B','C','D','E','F','G','H','I','J']
# actions = np.array(['B'])
# actions = np.array(['I_Love_You','OK','Name','My','How','Where','Hello','Father','Mother','Your'])
no_sequences = 30

sequence_length = 30







#
# import cv2
# import numpy as np
# import os
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Input
# from tensorflow.keras.models import Model
#
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
#
#
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results
#
#
# def draw_styled_landmarks(image, results):
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#
# def extract_keypoints(results):
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             rh = np.array(
#                 [[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(
#                 21 * 3)
#             return (np.concatenate([rh]))
#
#
# def build_cnn_lstm_model(input_shape, num_classes):
#     cnn_input = Input(shape=input_shape)
#     x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(cnn_input)
#     x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
#     x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
#     x = TimeDistributed(Flatten())(x)
#
#     x = LSTM(64, return_sequences=True, activation='relu')(x)
#     x = LSTM(128, return_sequences=True, activation='relu')(x)
#     x = LSTM(64, return_sequences=False, activation='relu')(x)
#
#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     output = Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs=cnn_input, outputs=output)
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#     return model
#
#
#
# # Paths and parameters
# DATA_PATH = os.path.join('MP_Data')
# actions = np.array(
#     ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
#      'Y', 'Z','I_Love_You','OK','Name','My','Hello','Father','Mother','Your'])
# # actions = np.array(['I_Love_You','OK','Name','My','How','Where','Hello','Father','Mother','Your'])
# no_sequences = 30
# sequence_length = 30
