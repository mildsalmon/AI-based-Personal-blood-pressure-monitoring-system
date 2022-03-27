# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

class BpMonitoringSystemByAi:
    def __init__(self):
        # seed 값 설정
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(3)

        """
        pandas, numpy, plt 초기 설정값 변경
        """
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 1000)
        # pd.set_option('display.width', 1000)
        #
        # np.set_printoptions(edgeitems=50)
        # np.set_printoptions(threshold=np.inf)
        # np.core.arrayprint._line_width = 180
        #
        # plt.rcParams['font.family'] = 'NanumSquare'

        self.epoch = 200
        self.batch_size = 5

    def rp_preprocess(self, file_name: str, info_dir: str) -> None:
        """
        라즈베리파이에서 수집한 데이터 전처리
        """

        """
        Extract
        """
        read_path = self.set_path(file_name)
        info_path = self.set_path(info_dir)

        """
        Transform
        """
        spo2_wave_first = [5, 70]

        select_wave_pd = pd.DataFrame()

        spo2_wave_all_avg_list = []
        select_wave_full_avg_list = []

        ppg_pd = self.load_collection_data(path=read_path)
        info_pd = self.load_collection_data(path=info_path)

        first_interval = ppg_pd.iloc[spo2_wave_first[0]: spo2_wave_first[1], :]

        first_ppg = first_interval.iloc[:, 0]
        first_index = first_ppg.astype(float).idxmax()

        ppg_pd = ppg_pd.iloc[first_index:, :]

        spo2_wave_all = 0

        for spo2_wave in ppg_pd.iloc[:, 0]:
            spo2_wave_all = spo2_wave + spo2_wave_all

        spo2_wave_all_avg = spo2_wave_all / len(ppg_pd.iloc[:, 0])

        min = ppg_pd.iloc[:, 0].max()
        min_1_count = 0
        j_over = 0

        for j in range(len(ppg_pd.iloc[:, 0])):
            if min > ppg_pd.iloc[j, 0]:
                min = ppg_pd.iloc[j, 0]

                if min_1_count > 0:
                    j_over = min_1_count
                    min_1_count = 0
            else:
                min_1_count = min_1_count + 1
                if min_1_count == 20:
                    min_1 = ppg_pd.iloc[:j, 0].min()
                    min_1_idx = ppg_pd.iloc[:j, 0].astype(float).idxmin()

                    min = ppg_pd.iloc[:, 0].max()
                    min_1_count = 0
                    j = j - 20 + j_over
                    break

        max = ppg_pd.iloc[:, 0].min()
        max_1_count = 0
        k_over = 0

        for k in range(len(ppg_pd.iloc[:, 0])):
            if max < ppg_pd.iloc[j + k, 0]:
                max = ppg_pd.iloc[j + k, 0]

                if max_1_count > 0:
                    k_over = max_1_count
                    max_1_count = 0
            else:
                max_1_count = max_1_count + 1

                if max_1_count == 10:
                    max_1 = ppg_pd.iloc[j: j+k, 0].max()
                    max_1_idx = ppg_pd.iloc[j: j+k, 0].astype(float).idxmax()

                    max = ppg_pd.iloc[:, 0].min()
                    max_1_count = 0
                    k = k - 10 + k_over
                    break

        for l in range(len(ppg_pd.iloc[:, 0])):
            if min > ppg_pd.iloc[j + k + l, 0]:
                min = ppg_pd.iloc[j + k + l, 0]

                if min_1_count > 0:
                    min_1_count = 0
            else:
                min_1_count = min_1_count + 1

                if min_1_count == 20:
                    min_2 = ppg_pd.iloc[j + k: j + k + l, 0].min()
                    min_2_idx = ppg_pd.iloc[j + k: j + k + l, 0].astype(float).idxmin()

                    min_1_count = 0
                    break

        max = ppg_pd.iloc[:, 0].max()
        min = ppg_pd.iloc[:, 0].min()

        one_wave_len = min_2_idx - min_1_idx

        start_point = [ppg_pd.index[0]]
        end_point = [start_point[0] + 127]

        x = range(len(ppg_pd.loc[:,'PPG']))
        """
        환자 감시 장치에서 2분동안 수집한 PPG 신호 전체 그래프
        """
        # plt.rcParams["figure.figsize"] = (18, 7)
        # plt.plot(x, ppg_pd.loc[:,'PPG'])
        # # plt.axis([0, len(ppg_pd.loc['SpO2 Wave']), ppg_pd.loc['SpO2 Wave'].min(),
        # #           ppg_pd.loc['SpO2 Wave'].max()])
        # # plt.savefig(plt_save_path + "\\{0}_1_Full.png".format(num_1_file_name))
        # # plt.show()
        # # plt.cla()

        """
        -250 ~ 250 박스
        """
        y = -250 + spo2_wave_all_avg
        y = [y for _ in x]

        plt.plot(x, y, c='k')

        y = 250 + spo2_wave_all_avg
        y = [y for _ in x]

        plt.plot(x, y, c='k')
        # plt.savefig(plt_save_path + '\\{0}_6_wave_all_avg_line.png'.format(num_1_file_name))
        # plt.show()
        # plt.cla()

        """
        파형 한개의 크기를 표시
        빨간색 점 = 시작하는 지점
        파란색 점 = 파형의 최고점
        초록색 점 = 파형의 끝점
        """
        # min 1
        plt.scatter(min_1_idx, min_1, c = 'r')

        # max 1
        plt.scatter(max_1_idx, max_1, c = 'b')

        # min 2
        plt.scatter(min_2_idx, min_2, c = 'g')

        color = ["b", "g", "r", "c", "m", "y"]

        for m in end_point:
            if m + one_wave_len < ppg_pd.index[-1]:
                check_point = m
                check_point_end = check_point + one_wave_len

                min_wave = ppg_pd.loc[check_point:check_point_end, 'PPG'].min()
                min_wave_idx = ppg_pd.loc[check_point:check_point_end, 'PPG'].astype(float).idxmin()

                check_point = min_wave_idx
                check_point_end = check_point + (one_wave_len // 1)
                max_wave = ppg_pd.loc[check_point:check_point_end, 'PPG'].max()
                max_wave_idx = ppg_pd.loc[check_point:check_point_end, 'PPG'].astype(float).idxmax()

                start_point.append(max_wave_idx)

                if max_wave_idx + 127 < ppg_pd.index[-1]:
                    end_point.append(max_wave_idx + 127)
                else:
                    end_point.append(ppg_pd.index[-1])

        # plt.plot(x, ppg_pd.loc[:, "PPG"], c='k')

        for n in range(len(start_point[:-1])):
            x = range(start_point[n], end_point[n] + 1)
            y = ppg_pd.loc[start_point[n]:end_point[n], 'PPG']

            plt.plot(x, y, c=color[n%len(color)])

        bio_index = []

        for p in range(128):
            bio_index.append(p)

        select_wave_one_avg_list = []

        for o, wave_start in enumerate(start_point[:-1]):
            select_wave_one = copy.deepcopy(ppg_pd.loc[wave_start:end_point[o], 'PPG'])

            height = info_pd.iloc[0, 0]
            weight = info_pd.iloc[0, 1]

            h_w_Seris = pd.Series({'height': height,
                                   'weight': weight})

            select_wave_one_one = 0

            for r in select_wave_one:
                select_wave_one_one = select_wave_one_one + r

            select_wave_one_avg = select_wave_one_one / len(select_wave_one)

            if max in select_wave_one.unique():
                continue
            if min in select_wave_one.unique():
                continue

            select_wave_one_avg_list.append(select_wave_one_avg)

            x = range(wave_start - start_point[0], wave_start + 128 - start_point[0])
            y = [select_wave_one_avg for _ in x]
            # plt.plot(x, select_wave_one, c='cornflowerblue')  # cornflowerblue #steelblue
            plt.plot(x, y, c=color[o % len(color)])

            for s in range(len(select_wave_one)):
                select_wave_one.iloc[s] = select_wave_one.iloc[s] - spo2_wave_all_avg

            for q in range(len(select_wave_one)):
                select_wave_one.iloc[q] = select_wave_one.iloc[q] / 1024

            select_wave_one.index = bio_index

            select_wave_one = pd.concat([h_w_Seris, select_wave_one])

            select_wave_pd = select_wave_pd.append(select_wave_one, ignore_index=True)

        plt.show()

        select_wave_full_avg_list.append(select_wave_one_avg_list)

        """
        Load
        """
        select_wave_pd.to_csv("rp_text.csv", mode='w', header=True)

    def learn(self, dir_name):
        """
        인공지능 모델 학습을 위한 함수
        :param dir_name:
            예측할 데이터가 있는 디렉터리명
            ex) data\\collection.csv
        :return:
        """

        path = self.set_path(dir_name)
        csv_data = self.load_collection_data(path=path)

        wave = csv_data.iloc[:,0:128]
        BP = csv_data.iloc[:,-4:-2]
        HW = csv_data.iloc[:,-2:]

        """
        PPG 정보를 list로 변환
        """
        wave_list = wave.values.tolist()
        """
        키, 몸무게를 각 변수로 분리
        """
        Height = HW.iloc[:, 0]
        Weight = HW.iloc[:, 1]

        Height = Height.values
        Weight = Weight.values

        """
        키, 몸무게(십진수 데이터)를 gray code로 변환
        """
        Height_gray_code_list = self.convert_DEC_to_GrayCode(Height)
        Weight_gray_code_list = self.convert_DEC_to_GrayCode(Weight)

        """
        분리한 키, 몸무게의 gray code를 합침
        """
        HW_gray_code_list = self.list_append(Height_gray_code_list, Weight_gray_code_list)

        """
        혈압 정보를 수축기 혈압과 이완기 혈압으로 구분
        """
        BP_D = BP.iloc[:, 0]
        BP_S = BP.iloc[:, 1]

        BP_D = BP_D.values
        BP_S = BP_S.values

        BP_D_gray_code_list = self.convert_DEC_to_GrayCode(BP_D)
        BP_S_gray_code_list = self.convert_DEC_to_GrayCode(BP_S)

        X_np = self.list_append(wave_list, HW_gray_code_list)
        Y_np = self.list_append(BP_D_gray_code_list, BP_S_gray_code_list)

        X_data = self.make_np_array(X_np)
        Y_data = self.make_np_array(Y_np)

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)#, random_state=seed)

        model, history = self.model(X_train, X_test, Y_train, Y_test)

        model.summary()

        print(model.evaluate(X_test, Y_test))

        print(history.history['val_loss'])
        print(history.history['loss'])

        print(model.predict(X_train[1:2], batch_size=self.batch_size))
        print(Y_train[1])

        predict = model.predict(X_test[:], batch_size=self.batch_size)
        Y_test_use = Y_test[:]

        print("X_test[1] : \n", X_test[0:10])
        print("model.predict(X_test[1:2], batch_size=5) : \n", predict)
        print("Y_test[1] : \n",Y_test_use)

        BP_D_dec, BP_S_dec, BP_D_Y_dec, BP_S_Y_dec, per = self.postprocess(predict, Y_test_use)

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
        plt.axis([-10, self.epoch+10, 0, 1])
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
        per = pd.DataFrame([per], index=['gray code per'])

        Y_dec_Series = Y_dec_Series.transpose()
        predict_dec_Serires = predict_dec_Serires.transpose()
        per = per.transpose()

        save_csv = pd.concat([X_test_Series, Y_dec_Series, predict_dec_Serires, per], axis=1)

        print("save_csv : ", save_csv)

        self.save_prediction_data(save_csv, "prediction_learn.csv")

    def model(self, X_train, X_test, Y_train, Y_test):
        """
        학습에 사용하는 인공지능 모델
        :param X_train:
        :param X_test:
        :param Y_train:
        :param Y_test:
        :return:
        """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(64, input_dim=144, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=self.epoch, batch_size=self.batch_size, validation_data=(X_test, Y_test))

        self.save_model(model)

        return model, history

    def predict(self, dir_name):
        """
        인공지능 모델 예측을 위한 함수
        기존에 학습한 모델 정보를 통해 새로운 데이터(PPG)의 혈압 정보(수축기 혈압, 이완기 혈압)를 예측함
        :param dir_name:
            예측할 데이터가 있는 디렉터리명
            ex) data\\unknown.csv
        :return:
        """
        path = self.set_path(dir_name)
        csv_data = self.load_collection_data(path=path)

        wave = csv_data.iloc[:,0:128]
        HW = csv_data.iloc[:,-2:]

        wave_list = wave.values.tolist()

        Height = HW.iloc[:, 0]
        Weight = HW.iloc[:, 1]

        Height = Height.values
        Weight = Weight.values

        Height_gray_code_list = self.convert_DEC_to_GrayCode(Height)
        Weight_gray_code_list = self.convert_DEC_to_GrayCode(Weight)

        HW_gray_code_list = self.list_append(Height_gray_code_list, Weight_gray_code_list)

        X_np = self.list_append(wave_list, HW_gray_code_list)

        X_data = self.make_np_array(X_np)

        model = self.load_model()

        model.summary()

        print(model.predict(X_data[1:2], batch_size=self.batch_size))

        predict = model.predict(X_data[:], batch_size=self.batch_size)

        print("X_test[1] : \n", X_data[0:10])
        print("model.predict(X_test[1:2], batch_size=5) : \n", predict)

        BP_D_dec, BP_S_dec = self.postprocess(predict)

        # print("\n====Y 값====\n")
        # print("BP_D_Y_dec", BP_D_Y_dec)
        # print("BP_S_Y_dec", BP_S_Y_dec)

        print("\n====predict 값====\n")
        print("BP_D_dec", BP_D_dec)
        print("BP_S_dec", BP_S_dec)

        for i in range(len(BP_D_dec)):
        #     print("\n====Y 값====\n")
        #     print("BP_D_Y_dec", BP_D_Y_dec[i])
        #     print("BP_S_Y_dec", BP_S_Y_dec[i])
        #
            print("\n====predict 값====\n")
            print("BP_D_dec", BP_D_dec[i])
            print("BP_S_dec", BP_S_dec[i])

        # print("정확도 : ", model.evaluate(X_data, Y_data)[1])
        # print("오차 : ", model.evaluate(X_data, Y_data)[0])

        return BP_D_dec, BP_S_dec

    def postprocess(self, predict) -> list:
        """
        후처리 과정
        :param predict:
            인공지능 모델링을 통해 얻은 순수 예측값
        :return:
            predction(수축기 혈압, 이완기 혈압), 실측 값(수축기 혈압, 이완기 혈압), 예측값과 실측값 사이의 정확도(%)
        """
        predict_list = self.convert_NP_to_LIST(predict)
        gray_list = self.convert_Pridiction_to_GrayCode(predict_list)
        BP_D_dec, BP_S_dec = self.convert_GrayCode_to_DEC(gray_list)

        return [BP_D_dec, BP_S_dec]

    def search_csv_file(self, file_list: list) -> list:
        """
        csv 파일 검색
        디렉토리에서 csv파일만 검색해서 반환
        :param file_list:
            파일 리스트가 들어있는 디렉토리
        :return:
            csv파일의 리스트
        """
        result = []

        for name in file_list:
            if 'csv' in name:
                result.append(name)

        return result

    def set_path(self, dir_name: str) -> str:
        """
        경로 설정을 위해 현재 코드를 실행하는 위치와 입력한 디렉터리 경로를 합쳐줌
        C:/data + Model = C:/data/Model
        :param dir_name:
            입력된 디렉터리 경로
        :return:
            합쳐진 디렉터리 경로
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)

        return path

    def save_model(self, model):
        """
        학습한 인공지능 모델 데이터를 저장함
        저장하는 정보는
            1. 나중에 모델을 재구성하기 위한 모델의 구성 정보
            2. 모델을 구성하는 각 뉴런들의 가중치
            3. 모델의 컴파일 정보(compile()이 호출된 경우) / optimizer, loss
            4. 재학습을 할 수 있도록 마지막 학습 상태
        :param model:
            인공지능 모델
        :return:
        """
        model.save('model_save.h5')

    def load_model(self):
        """
        미리 학습시킨 인공지능 모델 데이터를 불러옴

        :return:
        """
        model = load_model('model_save.h5')

        return model

    def save_prediction_data(self, save, file_name: str):
        """
        prediction값을 csv파일로 저장
        :param save:
            저장할 데이터
        :param file_name:
            파일 이름
            파일 경로는 ./data로 고정
        :return:
        """
        path = self.set_path(file_name)

        save.to_csv(path, mode='w', header=True)

    def load_collection_data(self, path: str):
        """
        수집데이터를 불러옴
        불러오는 데이터의 형식은 csv
        csv파일을 pandas형식으로 저장
        :param path:
            데이터가 저장된 위치
        :return:
            pandas 자료형 데이터
        """
        read_data = pd.read_csv(path, index_col=0)

        return read_data

    def match(self, real_data: list, predict_data: list) -> list:
        """
        검증데이터(Validation set)의 real Y값과 예측데이터(prediction set)의 prediction Y 사이의 유사도를 확인
        gray code(2진 코드)이기 때문에 동일한 위치에 동일한 값이 있는지를 확인
        :param real_data:
            검증데이터(Validation set)의 Y값
        :param predict_data:
            예측데이터(prediction set)의 Y값
        :return:
            유사도를 백분율로 나타낸 것들의 list
        """
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

    def make_np_array(self, data: list):
        """
        리스트 데이터를 텐서플로우에서 사용할 수 있게 numpy의 array로 변경
        :param data:
            리스트 자료형 데이터
        :return:
            array 자료형 데이터
        """
        result = np.array(data)

        return result

    def list_append(self, BP_D: list, BP_S: list) -> list:
        """
        두 개의 리스트를 하나로 만들기 위해 더하는 함수
        :param BP_D:
        :param BP_S:
        :return:
        """
        result = []

        for i,_ in enumerate(BP_D):
            result.append(BP_D[i] + BP_S[i])

        return result

    def convert_NP_to_LIST(self, np) -> list:
        """
        numpy의 array 형식의 데이터를 list로 변환
        :return:
            list 자료형의 값
        """
        list = np.tolist()

        return list

    def convert_Pridiction_to_GrayCode(self, prediction_list: list) -> list:
        """
        인공지능 모델을 통해 얻어진 prediction값은 0과 1로 이루어진 binary 형식의 값이 아니다.
        이 모델은 키와 몸무게를 gray code로 변환시켜서 학습했기 때문에 예측값도 binary 형식으로 바꿔주는 작업이 필요하다.
        :param prediction_list:
            인공지능 모델을 통해 얻어진 prediction 값
        :return:
            prediction 값을 gray code로 변환시킨 값
        """
        binary_list = []

        for predict in prediction_list:
            predict_max = max(predict)
            predict_min = min(predict)
            half = (predict_max - predict_min)/2

            predict_to_binary = []

            for element in predict:
                if element >= half:
                    bin = 1
                elif element < half:
                    bin = 0
                predict_to_binary.append(bin)
            binary_list.append(predict_to_binary)

        return binary_list

    """
    Gray code를 십진수로 Decoding하는 부분
    """
    def convert_GrayCode_to_DEC(self, gray_code_list: list) -> list:
        """
        gray code를 십진수로 변환
        gray code는 십진수로 변환할 때 (gray code -> binary code -> 십진수)의 과정을 거친다.
        :param gray_code_list:
            gray code 리스트 값
        :return:
            변환된 십진수 리스트 값
        """
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
            BP_D_binary_code.append(self.convert_GrayCode_to_BinaryCode(BP_D))
            BP_S_binary_code.append(self.convert_GrayCode_to_BinaryCode(BP_S))

        BP_D_dec_list = self.convert_BinaryCode_to_DEC(BP_D_binary_code)
        BP_S_dec_list = self.convert_BinaryCode_to_DEC(BP_S_binary_code)

        return BP_D_dec_list, BP_S_dec_list

    def convert_GrayCode_to_BinaryCode(self, gray: list) -> list:
        """
        gray code를 binary code로 변환
        :param gray:
            gray code 리스트 값
        :return:
            변환된 binary code 리스트 값
        """
        binary_code = []
        x_bit = 0

        for i, bit in enumerate(gray):
            if i == 0:
                binary_code.append(bit)
                x_bit = bit
            else:
                x_bit = x_bit ^ bit
                binary_code.append(x_bit)

        return binary_code

    def convert_BinaryCode_to_DEC(self, binary: list) -> list:
        """
        binary code를 십진수로 변환
        :param binary:
            binary code 리스트 값
        :return:
            변환된 십진수 리스트 값
        """
        dec_list = []

        for code in binary:
            code = list(map(str, code))
            binary_sum = "0b"+"".join(code)
            dec_list.append(int(binary_sum, 2))

        return dec_list

    """
    십진수를 Gray code로 Encoding하는 부분
    """
    def convert_DEC_to_GrayCode(self, dec: list) -> list:
        """
        Array를 graycode로 변환
        십진수 값을 graycode 값으로 변환한다.
        graycode는 (십진수 -> binary code -> gray code)의 과정을 거쳐야함
        :param dec:
            십진수 리스트 값
        :return:
            변환된 graycode 리스트 값
        """
        binary_BP_list = self.convert_DEC_to_BinaryCode(dec)

        Gray_BP_list = self.convert_BinaryCode_to_GrayCode(binary_BP_list)

        return Gray_BP_list

    def convert_BinaryCode_to_GrayCode(self, binary: list) -> list:
        """
        binary code를 gray code로 변환
        :param binary:
            binary code 리스트 값
        :return:
            변환된 gray code 리스트 값
        """
        Gray_BP_list = []

        for i, element in enumerate(binary):
            Gray_BP = []

            for j, code in enumerate(element):
                if j == 0:
                    Gray_BP.append(code)
                else:
                    value = element[j-1] ^ element[j]
                    Gray_BP.append(value)
            Gray_BP_list.append(Gray_BP)

        return Gray_BP_list

    def convert_DEC_to_BinaryCode(self, dec: list) -> list:
        """
        십진수를 gray code로 변환하기 위해 binary code로 변환
        십진수를 binary code로 변환하면서 8자리가 안되는 binary code는 앞에 0을 채워서 8자리를 맞춰줌
        :param dec:
            십진수 리스트 값
        :return:
            변환된 binary code 리스트 값
        """
        binary_BP_list = []

        for i, element in enumerate(dec):
            element = int(element)
            binary_BP = format(element, 'b')
            binary_BP_list.append(list(map(int, binary_BP)))
            if len(binary_BP_list[i]) < 8:
                for j in range(8-len(binary_BP_list[i])):
                    binary_BP_list[i].insert(0, 0)

        return binary_BP_list


if __name__ == "__main__":
    A = BpMonitoringSystemByAi()

    # A.rp_preprocess()
    # A.monitoring_preprocess("data/Collection", "data/info.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data\\unknown.csv")

    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data\\unknown.csv")

### 학습 부분에서 데이터를 트레이닝 셋과 테스트 셋으로 나눈 것

    # 학습 : 28개 수집 데이터
    # 예측 : 28개 검증 데이터
    # 정확도 : 0.9466
    # 오차 : 0.1451
    # A.monitoring_preprocess("data/Collection", "data/info.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data/collection_new.csv")

    # 학습 : 28개 수집 데이터
    # 예측 : 100개 검증 데이터
    # 정확도 : 0.7281
    # 오차 : 2.0251
    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data/collection_new.csv")

    # 학습 : 100개 검증 데이터
    # 예측 : 28개 수집 데이터
    # 정확도 : 0.6974
    # 오차 : 0.7929
    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")
    # A.learn("data/collection_new.csv")
    # A.predict("data/collection.csv")

    # 학습 : 100개 검증 데이터
    # 예측 : 100개 검증 데이터
    # 정확도 : 0.8289
    # 오차 : 0.3342
    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")
    # A.learn("data/collection_new.csv")
    # A.predict("data/collection_new.csv")

### 학습 부분에서 데이터를 전부 트레이닝 셋으로 만듬

    # 라즈베리파이 실험
    # A.rp_preprocess("data/ppg.csv", "data/info_rp.csv")
    # A.predict("data/rp_text.csv")

    # A.monitoring_preprocess("data/test", "data/info_rp.csv")
    # A.predict("data/collection_new.csv")