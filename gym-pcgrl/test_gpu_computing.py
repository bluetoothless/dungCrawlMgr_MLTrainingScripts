import tensorflow as tf
from tensorflow.python.client import device_lib

# Print available GPUs
local_device_protos = device_lib.list_local_devices()
gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
print("Num GPUs Available: ", len(gpus))

# Configure TensorFlow to use GPU memory efficiently
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Sample computation to demonstrate GPU usage
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(f"Result:\n{session.run(c)}")  # Execute the computation
