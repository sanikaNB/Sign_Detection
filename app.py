from Function import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# import keras.api.v2.keras as keras
#
# to_categorical = keras.utils.to_categorical
# model_from_json = keras.models.model_from_json
# TensorBoard = keras.callbacks.TensorBoard


try:
    from tensorflow.keras.models import model_from_json
    print("abc")
except ImportError as e:
    print(f"Error importing to_categorical: {e}")



json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("modelll.h5")

colors = []
for i in range(0, 20):
    colors.append((245, 117, 16))
# print(len(colors))


def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        # print(frame.shape)
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
        image, results = mediapipe_detection(cropframe, hands)
        # print(results)

        # Draw landmarks
        # draw_styled_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-28:]

        try:
            if len(sequence) == 28:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
        except Exception as e:
            # print(e)
            pass

        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#
#







# from Function import actions, sequence_length, mediapipe_detection
# import numpy as np
# import cv2
# import mediapipe as mp
# from tensorflow.keras.models import model_from_json
#
# # Load the model
# json_file = open("model.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("model.h5")
#
#
# # Function to visualize probabilities
# def prob_viz(res, actions, input_frame, colors, threshold):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#     return output_frame
#
#
# def extract_keypoints(results):
#     # Check if there are any hand landmarks detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             keypoints = []
#             for lm in hand_landmarks.landmark:
#                 keypoints.append([lm.x, lm.y, lm.z])
#             return np.array(keypoints).flatten()
#     return np.zeros(21 * 3)  # Assuming 21 landmarks per hand, return zero array if no landmarks found
#
#
# # Detection variables
# sequence = []
# sentence = []
# accuracy = []
# predictions = []
# threshold = 0.8
#
# cap = cv2.VideoCapture(0)
#
# # Set mediapipe model
# mp_hands = mp.solutions.hands
# with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         cropframe = frame[40:400, 0:300]
#         frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
#         image, results = mediapipe_detection(cropframe, hands)
#
#         # Debugging: Print results from mediapipe
#         print("Mediapipe results:", results)
#
#         keypoints = extract_keypoints(results)
#         # Debugging: Print extracted keypoints
#         print("Extracted keypoints:", keypoints)
#
#         sequence.append(keypoints)
#         sequence = sequence[-sequence_length:]
#
#         try:
#             if len(sequence) == sequence_length:
#                 # Reshape the sequence to match the model's expected input shape
#                 reshaped_sequence = np.expand_dims(sequence, axis=-1)  # Add a new dimension for the channel
#                 reshaped_sequence = reshaped_sequence.reshape(
#                     (1, sequence_length, 21, 3, 1))  # Reshape to (1, 30, 21, 3, 1)
#
#                 res = model.predict(reshaped_sequence)[0]
#                 # Debugging: Print model predictions
#                 print("Model prediction:", res)
#
#                 print(actions[np.argmax(res)])
#                 predictions.append(np.argmax(res))
#
#                 if np.unique(predictions[-10:])[0] == np.argmax(res):
#                     if res[np.argmax(res)] > threshold:
#                         if len(sentence) > 0:
#                             if actions[np.argmax(res)] != sentence[-1]:
#                                 sentence.append(actions[np.argmax(res)])
#                                 accuracy.append(str(res[np.argmax(res)] * 100))
#                         else:
#                             sentence.append(actions[np.argmax(res)])
#                             accuracy.append(str(res[np.argmax(res)] * 100))
#
#                 if len(sentence) > 1:
#                     sentence = sentence[-1:]
#                     accuracy = accuracy[-1:]
#
#         except Exception as e:
#             print("Error during prediction:", e)
#             pass
#
#         cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
#         cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (255, 255, 255), 2, cv2.LINE_AA)
#
#         cv2.imshow('OpenCV Feed', frame)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
