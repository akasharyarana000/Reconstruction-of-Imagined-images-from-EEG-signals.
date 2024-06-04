import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Static learning rate optimizer
optimizer_static = tf.keras.optimizers.Adam(learning_rate=5e-4)

# Exponential decay learning rate optimizer
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer_decay = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile and train models
model_static = create_model()
model_static.compile(optimizer=optimizer_static, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_static = model_static.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

model_decay = create_model()
model_decay.compile(optimizer=optimizer_decay, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_decay = model_decay.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Plot training and validation accuracy
plt.plot(history_static.history['val_accuracy'], label='Static Learning Rate')
plt.plot(history_decay.history['val_accuracy'], label='Exponential Decay Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig('learningrate.png')
plt.show()

