import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from layers import RBFLayer


N_POINTS = 32
BATCH_SIZE = 16

x_in = Input(shape=(1,))
h = RBFLayer(units=64, name='rbf')(x_in)
y_out = Dense(1, activation='linear')(h)

model = Model(inputs=x_in, outputs=y_out)
tf.keras.utils.plot_model(model, 'bp_ann.png', show_shapes=True, show_layer_names=True)

model.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.8),
    loss='mse',
    metrics=[]
)

tensorboard = TensorBoard(log_dir='logs')
earlystopping = EarlyStopping(monitor='loss', mode='min', patience=48, min_delta=1e-8)

x = np.linspace(-4, 4, N_POINTS)
y = 1.1 * (1 - x + 2 * (x ** 2)) * np.exp(-(x ** 2) / 2)
noise = np.random.normal(0., .1, np.shape(y))
y += noise
x = normalize(x)
y = normalize(y)

xy_data = tf.data.Dataset.from_tensor_slices((x, y))
train_data = xy_data.shuffle(32).batch(BATCH_SIZE)


model.fit(
    x=train_data,
    epochs=102400,
    callbacks=[tensorboard, earlystopping]
    # callbacks=[tensorboard]
)

y_pred = model(x).numpy()
plt.plot(x, y_pred, 'b')
plt.plot(x, y, 'r')
plt.show()

model.save_weights('rbf_weights.h5')
