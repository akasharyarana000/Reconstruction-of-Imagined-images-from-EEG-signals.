from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pathlib
import Template

current_folder = pathlib.Path(__file__).parent


# Parameters
input_row = 128  # Ch
input_column = 125  # Time
input_color = 1
category_size = 40
batch_size = 64
Epoch = 1000


# Parameter for Sinc Layer
n_filter = 16
filter_dim = 65  # Odd Number
multiplier = 2
sampling_rate = 250
frequency_scale = sampling_rate
min_freq = 1.0
min_band = 4.0
band_initial = 1.0  # Initial Sinc Band: min_band + band_initial
low_freq = 1.0 - min_freq
high_freq = 40.0 - min_freq
seed = 13579


#Instead of Static learning rate add a exponential decay learning rate
# Define the exponential decay learning rate schedule
initial_learning_rate = 5e-4
decay_steps = 10000  # Number of steps before decay
decay_rate = 0.96  # Decay rate

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True  # If True, learning rate decays at discrete intervals
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)





#optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
# Load Data
loader_output = Template.data_imagination40_loader_train_val_test_sub_all(do_norm=True, do_zscore=True)  # All subject's data

train_data = loader_output[0]
train_label = loader_output[1]  # Corrected assignment
validation_data = loader_output[2]
validation_label = loader_output[3]  # Corrected assignment
test_dat = loader_output[4]
test_label = loader_output[5]
counter = loader_output[6]

# Compiler
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Tensorboard
log_dir = 'Results/logs/fit/' + datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early Stop
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              min_delta=0.0,
                                              patience=10000)

# Train
start = time.time()
history = model.fit(train_data, train_label, epochs=Epoch, batch_size=batch_size,
                    validation_data=(validation_data, validation_label),
                    verbose=1,
                    callbacks=[tensorboard_callback, early_stop])

elapsed_time = time.time() - start
test_loss, test_accuracy = model.evaluate(test_dat, test_label)
print(f'Tested Accuracy: {test_accuracy}, Time: {elapsed_time} sec (Epoch: {Epoch})')

# Save Final Model and History
now = datetime.datetime.now()
now_time = now.strftime('%y%m%d%_H%M%S')
model_save_name = now_time + '_Feature_Extractor_Model'
model.save(current_folder/'Results'/model_save_name)

history_save_name = 'Results/' + now_time + '_Extractor_History'
np.savez(history_save_name,
         acc=history.history['accuracy'],
         loss=history.history['loss'],
         val_acc=history.history['val_accuracy'],
         val_loss=history.history['val_loss'])

# Plot
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('Training Imagine Loss.png')

