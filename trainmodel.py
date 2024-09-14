import Function
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


label_map = {label:num for num, label in enumerate(Function.actions)}
# print(label_map)
sequences, labels = [], []
for action in Function.actions:
    for sequence in range(Function.no_sequences):
        window = []
        for frame_num in range(Function.sequence_length):
            res = Function.np.load(Function.os.path.join(Function.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = Function.np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = Function.os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(Function.actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=30, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('modelll.h5')











# import Function
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import TensorBoard
#
# # Preparing data
# label_map = {label: num for num, label in enumerate(Function.actions)}
# sequences, labels = [], []
# for action in Function.actions:
#     for sequence in range(Function.no_sequences):
#         window = []
#         for frame_num in range(Function.sequence_length):
#             res = Function.np.load(Function.os.path.join(Function.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
#
# X = Function.np.array(sequences)
# y = to_categorical(labels).astype(int)
#
# # Reshape X to match the input shape for Conv2D
# # Reshape to (batch_size, sequence_length, height, width, channels)
# X = X.reshape((X.shape[0], X.shape[1], 21, 3, 1))  # 21 keypoints, 3 coordinates, 1 channel
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#
# # Building the model
# input_shape = (Function.sequence_length, 21, 3, 1)  # Adjust input shape as per your data
# model = Function.build_cnn_lstm_model(input_shape, len(Function.actions))
#
# # Training the model
# log_dir = Function.os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)
# model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
# model.summary()
#
# # Saving the model
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('model.h5')
#

