import tensorflow as tf

print(tf.__version__)
print(tf.test.is_gpu_available())
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
