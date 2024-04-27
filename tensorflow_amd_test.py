import tensorflow as tf

# Check if DirectML is available
if tf.test.is_built_with_direct_mkl():
    print("DirectML support is available.")
else:
    print("DirectML support is not available.")

# Force TensorFlow to use DirectML backend
tf.config.set_soft_device_placement(True)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('RX570')[0], True)  # Ensure memory growth

# Your TensorFlow code here
