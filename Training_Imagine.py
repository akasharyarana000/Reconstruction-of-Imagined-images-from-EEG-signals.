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
loader = Template.data_imagination40_loader_train_val_test_sub_all(do_norm=True, do_zscore=True)  # All subject's data
# loader = Template.data_imagination40_loader_train_val_test_sub001(do_norm=True, do_zscore=True)  # One subject's data
# loader = Template.data_imagination40_loader_train_val_test_sub002(do_norm=True, do_zscore=True)  # One subject's data
# loader = Template.data_imagination40_loader_train_val_test_sub003(do_norm=True, do_zscore=True)  # One subject's data
# loader = Template.data_imagination40_loader_train_val_test_sub004(do_norm=True, do_zscore=True)  # One subject's data

train_data = loader[0]
# train_vis_features = loader[1]
train_label = loader[2]
# train_image = loader[3]
validation_data = loader[4]
# validation_vis_features = loader[5]
validation_label = loader[6]
# validation_image = loader[7]
test_dat = loader[8]
# test_vis_features = loader[9]
test_label = loader[10]
counter = loader[11]

print(f'Data: {train_data.shape}')
print(validation_data.shape)
print(test_dat.shape)
print(counter)


def sinc(band, t_right):
    y_right = K.sin(2*np.pi*band*t_right)/(2*np.pi*band*t_right)
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, [1.0], y_right])
    return y


class SincConv(Layer):

    def __init__(self, n_filter, filter_dim, sr, freq_scale, **kwargs):
        self.n_filter = n_filter
        self.filter_dim = filter_dim
        self.sr = sr
        self.freq_scale = freq_scale

        super(SincConv, self).__init__(**kwargs)

    def get_config(self):
        return {'n_filter': self.n_filter,
                'filter_dim': self.filter_dim,
                'sr': self.sr,
                'freq_scale': self.freq_scale}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.filter_b1 = self.add_weight(name='filter_b1',
                                         shape=(self.n_filter,),
                                         initializer='uniform',
                                         trainable=True)
        self.filter_band = self.add_weight(name='filter_band',
                                           shape=(self.n_filter,),
                                           initializer='uniform',
                                           trainable=True)

        np.random.seed(seed)
        initial_b1 = np.random.uniform(low_freq, high_freq, n_filter)
        initial_band = np.zeros_like(initial_b1) + band_initial

        self.set_weights([initial_b1 / self.freq_scale, initial_band / self.freq_scale])

        # Hamming
        n = np.linspace(0, self.filter_dim, self.filter_dim)
        window = 0.54 - 0.46 * K.cos(2 * np.pi * n / self.filter_dim)
        window = K.cast(window, 'float32')
        self.window = K.constant(window, name='window')

        t_right_linspace = np.linspace(1, (self.filter_dim - 1) / 2, int((self.filter_dim - 1) / 2))
        self.t_right = K.constant(t_right_linspace / self.sr, name='t_right')

        super(SincConv, self).build(input_shape)

    def call(self, x, **kwargs):
        self.filter_begin_freq = K.abs(self.filter_b1) + min_freq / self.freq_scale
        self.filter_end_freq = K.clip(self.filter_begin_freq + K.abs(self.filter_band) + min_band / self.freq_scale,
                                      min_freq / self.freq_scale, (self.sr/2) / self.freq_scale)
        filter_list = []
        for i in range(self.n_filter):
            low_pass1 = 2 * self.filter_begin_freq[i] * sinc(self.filter_begin_freq[i] * self.freq_scale, self.t_right)
            low_pass2 = 2 * self.filter_end_freq[i] * sinc(self.filter_end_freq[i] * self.freq_scale, self.t_right)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / K.max(band_pass)
            filter_list.append(band_pass * self.window)
        filters = K.stack(filter_list)  # (out_channels, filter_width)
        filters = K.transpose(filters)  # (filter_width, out_channels)
        filters = K.reshape(filters, (1, self.filter_dim, 1, self.n_filter))  # (1, filter_width, 1, out_channels)

        out = K.conv2d(x, kernel=filters, padding='same')

        return out


# Model
def make_model():
    inputs = Input(shape=(input_row, input_column, input_color))
    x = SincConv(n_filter, filter_dim, sampling_rate, frequency_scale)(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D((input_row, 1), strides=(1, 1), padding='valid', depth_multiplier=multiplier,
                               use_bias=False, depthwise_constraint=tf.keras.constraints.max_norm(1.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.SeparableConv2D(n_filter*multiplier, (1, 8), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 3))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    features512 = layers.Dense(512, activation='relu', name='features512')(x)
    x = layers.Dropout(0.5)(features512)
    features2048 = layers.Dense(2048, activation='relu', name='features2048')(x)
    x = layers.Dropout(0.5)(features2048)
    outputs = layers.Dense(category_size, activation='softmax', name='outputs',
                           kernel_constraint=tf.keras.constraints.max_norm(0.25))(x)
    model0 = Model(inputs=inputs, outputs=outputs)
    return model0


model = make_model()


# Compiler
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# Save Model every 100 Epochs
class EpochSave(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 100 == 0 and epoch+1 != Epoch:
            now0 = datetime.datetime.now()
            now_time0 = now0.strftime(f'%y%m%d%_H%M%S')
            model_save_name0 = now_time0 + '_Feature_Extractor_Model'
            model.save(current_folder / 'Results' / model_save_name0)


# Tensorboard
log_dir = 'Results/logs/fit/' + datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Early Stop
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              min_delta=0.0,
                                              patience=10000)


# Train
list_of_frequency = []
filter_b1 = model.layers[1].weights[0].numpy() * frequency_scale
filter_band = model.layers[1].weights[1].numpy() * frequency_scale
for k in range(filter_b1.shape[0]):
    filter_begin_freq = np.absolute(filter_b1[k]) + min_freq
    filter_end_freq = filter_begin_freq + np.absolute(filter_band[k]) + min_band
    list_of_frequency.append([filter_begin_freq, filter_end_freq])

second_kernel = model.layers[3].depthwise_kernel.numpy()
second_kernel = np.transpose(second_kernel, (2, 3, 1, 0))
second_kernel = second_kernel.reshape((n_filter*multiplier, 128))


'''
# PLot of initial Sinc layer weights
for i in range(filter_b1.shape[0]):
    plt.plot(list_of_frequency[i], [i+1, i+1], linewidth=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('# of Filter')
plt.show()
for i in range(n_filter):
    plt.subplot(4, 4, i+1)
    plt.plot(second_kernel[2*i], label='1')
    plt.plot(second_kernel[2*i+1], label='2')
    plt.legend()
plt.show()
'''


start = time.time()
history = model.fit(train_data, train_label, epochs=Epoch, batch_size=batch_size,
                    validation_data=(validation_data, validation_label),
                    verbose=1) 

"""history = model.fit(train_data, train_label, epochs=Epoch, batch_size=batch_size,
                    validation_data=(validation_data, validation_label),
                    verbose=1,
                    callbacks=[tensorboard_callback, early_stop, EpochSave()])"""
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
         val_loss=history.history['val_loss'],
         initial_band=np.array(list_of_frequency),
         initial_depthwise=second_kernel)


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
plt.show()
