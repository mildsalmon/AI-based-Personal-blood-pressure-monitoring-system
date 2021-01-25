import tensorflow as tf
import pandas as pd
import numpy as np

path = "data/people.csv"
data = pd.read_csv(path)

print(data)

Y_data = data.iloc[:, 1:3]
X_data = data.iloc[:, 3:]

print(Y_data)
print(Y_data.shape)

print(X_data)
print(X_data.shape)

Y_data = Y_data.values
X_data = X_data.values

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_dim=256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_data, Y_data, epochs=200, batch_size=10)

print(model.evaluate(X_data, Y_data))