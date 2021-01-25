import tensorflow as tf
import pandas as pd
# import keras

path = "data/people.csv"
data = pd.read_csv(path)

print(data)

Y_data = data.iloc[:, 1:3]
X_data = data.iloc[:, 3:]

print(Y_data)
print(Y_data.shape)

print(X_data)
print(X_data.shape)

X = tf.keras.layers.Input(shape=[256])
Y = tf.keras.layers.Dense(2)(X)

model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(X_data, Y_data, epochs=30000)

print(model.predict(X_data[1:10]))
print(Y_data[1:10])

print(model.get_weights())


