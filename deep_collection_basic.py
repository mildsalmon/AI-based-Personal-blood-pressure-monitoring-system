import pandas as pd
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

class deep_collection_basic:
    def __init__(self):
        # seed 값 설정
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(3)

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)

        np.set_printoptions(edgeitems=50)
        # np.core.arrayprint._line_width = 180

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\collection.csv")

        print(path)

        csv_data = self.read_csv(path=path)
        print(csv_data)

        wave = csv_data.iloc[:,0:128]
        BP = csv_data.iloc[:,-4:-2]
        HW = csv_data.iloc[:,-2:]

        print(wave,"\n",BP, "\n", HW)

        wave_list = wave.values.tolist()

        print("wave_list", wave_list)
        # print("wave_list", *wave_list, sep='\n')
        print("wave_list[0]_len", len(wave_list[0]))
        print("wave_list_len", len(wave_list))
        print("wave_list_type", type(wave_list))

        Height = HW.iloc[:, 0]
        Weight = HW.iloc[:, 1]

        Height = Height.values
        Weight = Weight.values

        Height_gray_code_list = self.gray_code(Height)
        Weight_gray_code_list = self.gray_code(Weight)

        print("Height_gray_code_list", Height_gray_code_list)
        print("Weight_gray_code_list", Weight_gray_code_list)

        HW_gray_code_list = self.list_append(Height_gray_code_list, Weight_gray_code_list)

        print("HW_gray_code_list", HW_gray_code_list)

        BP_D = BP.iloc[:, 0]
        BP_S = BP.iloc[:, 1]

        BP_D = BP_D.values
        BP_S = BP_S.values

        print(type(BP_D))

        BP_D_gray_code_list = self.gray_code(BP_D)
        BP_S_gray_code_list = self.gray_code(BP_S)

        print("BP_D_gray_code_list", BP_D_gray_code_list)
        print("BP_S_gray_code_list", BP_S_gray_code_list)
        # BP_D_series = pd.Series(BP_D_gray_code_list)
        # BP_S_series = pd.Series(BP_S_gray_code_list)

        # BP_pd = pd.DataFrame([BP_D_series, BP_S_series])

        # print(BP_D_gray_code_list)
        # print(type(BP_D_gray_code_list))
        # print(BP_D_gray_code_list.shape)

        # X_data = wave.values.astype(float)
        # X = pd.concat([wave, HW], axis=1)
        # print("X:",X)

        X_np = self.list_append(wave_list, HW_gray_code_list)
        Y_np = self.list_append(BP_D_gray_code_list, BP_S_gray_code_list)

        X_data = self.make_np_array(X_np)
        Y_data = self.make_np_array(Y_np)

        print("X_data:",X_data)
        print("Y_data:",Y_data)

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)#, random_state=seed)

        self.modeling(X_train, X_test, Y_train, Y_test)

    def modeling(self, X_train, X_test, Y_train, Y_test):
        model = tf.keras.models.Sequential()

        # X_train_len = len(X_train)
        # model.add(tf.keras.layers.Embedding(X_train_len, 144))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1))
        # model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
        # model.add(tf.keras.layers.LSTM(32, activation='relu'))
        # model.add(tf.keras.layers.Dense(16))
        # model.add(tf.keras.layers.Activation('sigmoid'))
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])
        # history = model.fit(X_train, Y_train, epochs=200, batch_size=100, validation_data=(X_test, Y_test))

        # X_train_len = len(X_train)
        # # print(X_train_len)
        # model.add(tf.keras.layers.Embedding(X_train_len, 144))
        # model.add(tf.keras.layers.LSTM(64, activation='relu'))
        # model.add(tf.keras.layers.Dense(16, activation='softmax'))
        # model.compile(loss='binary_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])
        # history = model.fit(X_train, Y_train, epochs=20, batch_size=100,validation_data=(X_test, Y_test))

        model.add(tf.keras.layers.Dense(64, input_dim=144, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        # model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        # model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=200, batch_size=5, validation_data=(X_test, Y_test))

        model.summary()
        print(model.evaluate(X_test, Y_test))

        print(history.history['val_loss'])
        print(history.history['loss'])

        # print(X_train[1:10])
        print(model.predict(X_train[1:2], batch_size=5))
        print(Y_train[1])

        print("X_test[1]\n", X_test[1])
        print("model.predict(X_test[1:2], batch_size=5)\n",model.predict(X_test[1:2], batch_size=5))
        print("Y_test[1]\n",Y_test[1])

        print("정확도 : ", model.evaluate(X_test, Y_test)[1])
        print("오차 : ", model.evaluate(X_test, Y_test)[0])
        # print(X_test)
        # print(model.get_weights())

    def make_np_array(self, data):
        result = np.array(data)

        return result

    def list_append(self, BP_D, BP_S):
        result = []

        # result.extend(BP_D)
        # result.extend(BP_S)

        # print(type(BP_D))
        # print(type(BP_S))

        for i,_ in enumerate(BP_D):
            # print(BP_D[i])
            result.append(BP_D[i] + BP_S[i])

        # print("list_append result", *result, sep="\n")
        # print(len(result))

        # result = np.array(result)
        # print("list_append result", result)

        return result

    def read_csv(self, path):
        read_data = pd.read_csv(path, index_col=0)

        return read_data

    def make_list(self, data):
        data_list = []



    def gray_code(self, BP):
        #
        binary_BP_list = self.binary_code(BP)

        # print(binary_BP_list)

        Gray_BP_list = []

        for i, element in enumerate(binary_BP_list):
            Gray_BP = []

            for j, code in enumerate(element):
                if j == 0:
                    Gray_BP.append(code)
                else:
                    value = element[j-1] ^ element[j]
                    Gray_BP.append(value)
            # print(Gray_BP)
            Gray_BP_list.append(Gray_BP)
        # print(Gray_BP_list)

        # Gray_BP_np = np.array(Gray_BP_list)

        return Gray_BP_list

    def binary_code(self, BP):
        binary_BP_list = []

        for i, element in enumerate(BP):
            element = int(element)
            binary_BP = format(element, 'b')
            # print(element, "&", binary_BP)
            binary_BP_list.append(list(map(int, binary_BP)))
            # print(binary_BP_list)
            if len(binary_BP_list[i]) < 8:
                for j in range(8-len(binary_BP_list[i])):
                    binary_BP_list[i].insert(0, 0)

            # print(binary_BP_list)
        # print(len(binary_BP_list))
        # print(binary_BP_list)

        return binary_BP_list

    # def XOR(self, num1, num2):
    #     num1

if __name__ == "__main__":
    deep_collection_basic()