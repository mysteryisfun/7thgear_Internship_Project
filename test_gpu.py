# ~\anaconda3\envs\py310\python.exe test_gpu.py
#still doesnt use gpu
import tensorflow as tf
print(tf.config.list_logical_devices('GPU'))
print(tf.config.list_physical_devices('GPU'))