import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import TensorBoard

# Check TensorFlow version
print(tf.__version__)

# Test to_categorical function
print(to_categorical([0, 1, 2, 3], num_classes=4))

# Create a simple model and save it to JSON
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(10,))])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Load the model from JSON
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Set up TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs")
print("Setup successful")
