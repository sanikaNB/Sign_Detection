import Function
from time import sleep

for action in Function.actions:
    for sequence in range(Function.no_sequences):
        try:
            Function.os.makedirs(Function.os.path.join(Function.DATA_PATH, action, str(sequence)))
        except:
            pass

# cap = cv2.VideoCapture(0)
# Set mediapipe modeldata
with Function.mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # NEW LOOP
    # Loop through actions
    for action in Function.actions:
        # Loop through sequences aka videos
        for sequence in range(Function.no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(Function.sequence_length):

                # Read feed
                # ret, frame = cap.read()
                frame = Function.cv2.imread('Image/{}/{}.png'.format(action, sequence))
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # Make detections
                image, results = Function.mediapipe_detection(frame, hands)
                #                 print(results)

                # Draw landmarks
                Function.draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    Function.cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                         Function.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, Function.cv2.LINE_AA)
                    Function.cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                         Function.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, Function.cv2.LINE_AA)
                    # Show to screen
                    Function.cv2.imshow('OpenCV Feed', image)
                    Function.cv2.waitKey(200)
                else:
                    Function.cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                         Function.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, Function.cv2.LINE_AA)
                    # Show to screen
                    Function.cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = Function.extract_keypoints(results)
                npy_path = Function.os.path.join(Function.DATA_PATH, action, str(sequence), str(frame_num))
                Function.np.save(npy_path, keypoints)

                # Break gracefully
                if Function.cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # cap.release()
    Function.cv2.destroyAllWindows()