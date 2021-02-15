import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)


path = "data/people.csv"
data = pd.read_csv(path)

print(data)

Y_data = data.iloc[:, 1:3]
X_data = data.iloc[:, 3:]

print(Y_data)
print(Y_data.shape)
print(type(Y_data))

print(X_data)
print(X_data.shape)
print(type(X_data))

Y_data = Y_data.values
X_data = X_data.values

print(Y_data)
print(Y_data.shape)
print(type(Y_data))

print(X_data[1])
print(X_data.shape)
print(type(X_data))

X_data[X_data == 0] = 0.0
print(X_data[1])
print(type(X_data[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=seed)

model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Dense(128, input_dim=256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(8, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(4, activation='relu'))
# model.add(tf.keras.layers.Dense(4, input_dim=256, activation='relu'))
model.add(tf.keras.layers.Dense(4, input_dim=256, activation=tf.keras.layers.LeakyReLU(0.2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.layers.LeakyReLU(0.2)))
# model.add(tf.keras.layers.Dense(2, activation='relu'))
# model.add(tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(0.2)))
# model.compile(loss='mean_squared_error',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.compile(loss='mean_absolute_percentage_error',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.compile(loss='mean_absolute_error',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['accuracy'])

model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adam',
              metrics=['accuracy'])
#
# model.compile(loss='mean_absolute_percentage_error',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.compile(loss='mean_absolute_error',
#               optimizer='adam',
#               metrics=['accuracy'])

model.summary()
# model.fit(X_data, Y_data, epochs=200, batch_size=10)
# model.fit(X_train, Y_train, epochs=200, batch_size=5)
history = model.fit(X_train, Y_train, epochs=200, batch_size=5, validation_data=(X_test, Y_test))

print(model.evaluate(X_test, Y_test))

print(history.history['val_loss'])
print(history.history['loss'])

print((X_train[1:10]))
print(model.predict(X_train[1:10], batch_size=5))
print(Y_train[1:10])

print((X_test[1:10]))
print(model.predict(X_test[1:10], batch_size=5))
print(Y_test[1:10])

print(model.get_weights())