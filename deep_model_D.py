import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        np.set_printoptions(threshold=np.inf)
        np.core.arrayprint._line_width = 180

        plt.rcParams['font.family'] = 'NanumSquare'

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

        epoch = 200
        batch_size = 5

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
        history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_test, Y_test))

        model.summary()
        print(model.evaluate(X_test, Y_test))

        print(history.history['val_loss'])
        print(history.history['loss'])

        # print(X_train[1:10])
        print(model.predict(X_train[1:2], batch_size=batch_size))
        print(Y_train[1])

        predict = model.predict(X_test[:], batch_size=batch_size)
        Y_test_use = Y_test[:]

        print("X_test[1] : \n", X_test[0:10])
        print("model.predict(X_test[1:2], batch_size=5) : \n", predict)
        # print("type(predict) : \n", type(predict))
        print("Y_test[1] : \n",Y_test_use)

        predict_list = self.np_to_list(predict)
        Y_test_use_list = self.np_to_list(Y_test_use)

        gray_list = self.predict_to_gray(predict_list)
        Y_test_use_gray_list = self.predict_to_gray(Y_test_use_list)

        # print("predict_list :", predict_list)
        print("gray_list :\n", *gray_list, sep='\n')

        print("matching per : ", self.matching_per(self.np_to_list(Y_test_use), gray_list))

        BP_D_binary_code, BP_S_binary_code = self.binary_code(gray_list)
        # print("BP_D_binary_code :\n", *BP_D_binary_code, sep='\n')
        # print("BP_S_binary_code :\n", *BP_S_binary_code, sep='\n')

        BP_D_Y_test, BP_S_Y_test = self.binary_code(Y_test_use_gray_list)

        BP_D_dec = self.binary_to_dec(BP_D_binary_code)
        BP_S_dec = self.binary_to_dec(BP_S_binary_code)

        BP_D_Y_dec = self.binary_to_dec(BP_D_Y_test)
        BP_S_Y_dec = self.binary_to_dec(BP_S_Y_test)

        print("\n====Y 값====\n")
        print("BP_D_Y_dec", BP_D_Y_dec)
        print("BP_S_Y_dec", BP_S_Y_dec)

        print("\n====predict 값====\n")
        print("BP_D_dec", BP_D_dec)
        print("BP_S_dec", BP_S_dec)

        for i in range(len(BP_D_dec)):
            print("\n====Y 값====\n")
            print("BP_D_Y_dec", BP_D_Y_dec[i])
            print("BP_S_Y_dec", BP_S_Y_dec[i])

            print("\n====predict 값====\n")
            print("BP_D_dec", BP_D_dec[i])
            print("BP_S_dec", BP_S_dec[i])

        print("정확도 : ", model.evaluate(X_test, Y_test)[1])
        print("오차 : ", model.evaluate(X_test, Y_test)[0])
        # print(X_test)
        # print(model.get_weights())

        y_loss = history.history['loss']
        y_vloss = history.history['val_loss']
        y_acc = history.history['accuracy']
        y_vacc = history.history['val_accuracy']
        x_len = np.arange(len(y_acc))

        """
        v = 테스트셋에 대한 실험 결과
        v 없는 것 = 학습셋에 대한 실험 결과
        """
        plt.plot(x_len, y_vloss, c='red', label='테스트셋에 대한 결과')
        plt.plot(x_len, y_vacc, c='red')
        plt.plot(x_len, y_loss, c='blue', label='학습셋에 대한 결과')
        plt.plot(x_len, y_acc, c='blue')
        plt.legend(loc='center right')
        plt.axis([-10, epoch+10, 0, 1])
        plt.xlabel("epoch")
        plt.ylabel("정확도, 오차율")
        plt.title("테스트셋과 학습셋의 정확도와 오차율")
        plt.show()

        plt.plot(x_len, y_vloss, c='red', label='테스트셋에 대한 결과')
        plt.plot(x_len, y_loss, c='blue', label='학습셋에 대한 결과')
        plt.legend(loc='center right')
        plt.xlabel("epoch")
        plt.ylabel("오차율")
        plt.title("테스트셋과 학습셋의 오차율 비교")
        plt.show()

        plt.plot(x_len, y_vacc, c='red', label='테스트셋에 대한 결과')
        plt.plot(x_len, y_acc, c='blue', label='학습셋에 대한 결과')
        plt.legend(loc='center right')
        plt.xlabel("epoch")
        plt.ylabel("정확도")
        plt.title("테스트셋과 학습셋의 정확도 비교")
        plt.show()

        X_test_Series = pd.DataFrame(X_test)
        Y_test_Series = pd.DataFrame(Y_test)
        Y_dec_Series = pd.DataFrame([BP_D_Y_dec, BP_S_Y_dec], index=['real_BP_D', 'real_BP_S'])
        predict_dec_Serires = pd.DataFrame([BP_D_dec, BP_S_dec], index=['predict_BP_D', 'predict_BP_S'])
        per = pd.DataFrame([self.matching_per(self.np_to_list(Y_test_use), gray_list)], index=['gray code per'])

        Y_dec_Series = Y_dec_Series.transpose()
        predict_dec_Serires = predict_dec_Serires.transpose()
        per = per.transpose()

        save_csv = pd.concat([X_test_Series, Y_dec_Series, predict_dec_Serires, per], axis=1)

        # print(X_test_Series)
        # print(X_test_Series.shape)
        # print(Y_test_Series)
        # print(Y_test_Series.shape)
        # print(Y_dec_Series)
        # print(Y_dec_Series.shape)
        # print(predict_dec_Serires)
        # print(predict_dec_Serires.shape)

        print("save_csv : ", save_csv)

        self.save_list_as_csv(save_csv)

    def save_list_as_csv(self, save):
        save.to_csv("data/prediction.csv", mode='w', header=True)

    def matching_per(self, real_data, predict_data):
        if len(real_data[0]) != len(predict_data[0]):
            return
        else:
            real_data_len = len(real_data[0])
            per_list = []

            for i, single_list in enumerate(real_data):
                true_count = 0
                for j, element in enumerate(single_list):
                    if element == predict_data[i][j]:
                        true_count = true_count + 1
                    per = true_count / real_data_len
                per_list.append(per)

            return per_list

    def np_to_list(self, np, dtype=None):
        list = np.tolist()

        return list

    def predict_to_gray(self, predict_list):
        binary_list = []

        for predict in predict_list:
            predict_max = max(predict)
            predict_min = min(predict)

            # print(predict)
            # print(predict_max)
            # print(predict_min)

            half = (predict_max - predict_min)/2

            # print(half)

            predict_to_binary = []

            for element in predict:
                if element > half:
                    bin = 1
                elif element < half:
                    bin = 0
                # if element > 0:
                #     bin = 1
                # elif element <= 0:
                #     bin = 0
                predict_to_binary.append(bin)
            binary_list.append(predict_to_binary)

        return binary_list

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

    def binary_to_dec(self, binary_code):
        dec_list = []
        # binary_sum = []

        for code in binary_code:
            code = list(map(str, code))
            binary_sum = "0b"+"".join(code)
            dec_list.append(int(binary_sum, 2))
            # binary_sum = list(map(int, binary_sum))
            # dec_list.append(format(binary_sum, 'd'))

            # binary_sum.append("".join(code))

        # binary_sum = list(map(int, binary_sum))

        # for binary in binary_sum:
        #     dec_list.append(format(binary, 'd'))

        return dec_list

    def binary_code(self, gray_code_list):
        BP_D_binary_code = []
        BP_S_binary_code = []

        for i, gray_code in enumerate(gray_code_list):
            BP_D = []
            BP_S = []
            for j, code in enumerate(gray_code):
                if j < 8:
                    BP_D.append(code)
                else:
                    BP_S.append(code)
            BP_D_binary_code.append(self.gray_to_binary(BP_D))
            BP_S_binary_code.append(self.gray_to_binary(BP_S))

        return BP_D_binary_code, BP_S_binary_code

    def gray_to_binary(self, gray_code):
        binary_code = []
        x_bit = 0

        for i, bit in enumerate(gray_code):
            if i == 0:
                binary_code.append(bit)
                x_bit = bit
            else:
                x_bit = x_bit ^ bit
                binary_code.append(x_bit)

        return binary_code

    def gray_code(self, BP):
        #
        binary_BP_list = self.dec_to_binary_code(BP)

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

    def dec_to_binary_code(self, BP):
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