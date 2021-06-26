# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

class DistinguishingPeopleByPpgSignals:
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

        self.epoch = 2000
        self.batch_size = 5

    def monitoring_preprocess(self, dir_name, info_dir):
        """
        :param dir_name:
            전처리할 데이터가 들어 있는 딕셔너리명
            ex) data/Collection
        :param info_dir
            키, 몸무게가 들어 있는 csv 명
            ex) data/info.csv
        :return:
        """
        read_path = self.set_path(dir_name)
        info_path = self.set_path(info_dir)
        # print(read_path)

        list_dir = os.listdir(read_path)
        read_path_list = self.search_csv_file(list_dir)

        # print(read_path_list)

        """
        spo2 시작할때 발생하는 이상치를 제거하는 작업
        학습에 사용할 데이터의 시작지점을 대략적으로 표시한 것
        수동 작업 -> 추후 자동화 예정
        """
        spo2_wave_first = [0, 60]

        """
        1개의 waveform에서 학습을 위해 시작하는 지점(1개의 waveform에서 최고점)을 선택해서 spo2_wave_start 리스트에 추가함
        """
        spo2_wave_start = []
        """
        128개로 sampling한 waveform, 키, 몸무게를 한 묶음으로 데이터프레임에 추가함
        """
        select_wave_pd = pd.DataFrame()

        spo2_wave_all_avg_list = []
        select_wave_full_avg_list = []
        end = []

        for i, path in enumerate(read_path_list):
            num_1_file_name = path

            # print(num_1_file_name)

            num_1_path = read_path + "\\" + num_1_file_name

            num_1_pd = pd.read_csv(num_1_path)
            info_pd = pd.read_csv(info_path)

            # print(num_1_pd)

            num_1_pd_info = num_1_pd.info()

            # print(num_1_pd_info)
            # print(num_1_pd.index)

            # print(type(num_1_pd[['SpO2 Wave', ' SpO2', ' BP_S', ' BP_D', ' TIME']]))
            num_1_use = num_1_pd[[' SpO2', 'SpO2 Wave', ' BP_S', ' BP_D', ' TIME']]
            info_use = info_pd[['num']]

            # print(num_1_use)

            num_1_use_tp = num_1_use.transpose()

            # print(num_1_use_tp)

            # print(num_1_use_tp.loc['SpO2 Wave'].min())
            # print(num_1_use_tp.loc['SpO2 Wave'].max())

            x = range(len(num_1_use_tp.loc['SpO2 Wave']))
            # print(len(num_1_use_tp.loc['SpO2 Wave']))

            plt_save_path = self.set_path("image\\collection")
            # print(plt_save_path)

            """
            환자 감시 장치에서 2분동안 수집한 PPG 신호 전체
            """
            # plt.rcParams["figure.figsize"] = (18, 7)
            # plt.plot(x, num_1_use_tp.loc['SpO2 Wave'])
            # plt.axis([0, len(num_1_use_tp.loc['SpO2 Wave']), num_1_use_tp.loc['SpO2 Wave'].min(),
            #           num_1_use_tp.loc['SpO2 Wave'].max()])
            # plt.savefig(plt_save_path + "\\{0}_1_Full.png".format(num_1_file_name))
            # plt.show()
            # plt.cla()

            """
            학습에 사용할 앞에 1분을 제외한 부분을 빨간색으로 표시
            뒤에 1분은 cuff식으로 혈압을 재느라 PPG 신호가 변조됨
            """

            # x1 = range(4601)
            # x2 = range(4600, len(num_1_use_tp.loc['SpO2 Wave']) - 1)
            #
            # plt.plot(x1, num_1_use_tp.iloc[1, 0:4601])
            # plt.plot(x2, num_1_use_tp.iloc[1, 4600:len((num_1_use_tp.loc['SpO2 Wave'])) - 1], 'r')
            # plt.axis([0, len(num_1_use_tp.loc['SpO2 Wave']), num_1_use_tp.loc['SpO2 Wave'].min(),
            #           num_1_use_tp.loc['SpO2 Wave'].max()])
            # plt.savefig(plt_save_path + '\\{0}_2_Full_Red.png'.format(num_1_file_name))
            # plt.show()
            # plt.cla()

            choice_num_1_use = num_1_use_tp.iloc[:, :4600]

            print("cho", choice_num_1_use)

            """
            뒤에 1분 정도의 PPG(빨간색으로 표시한 부분)을 제외한 나머지 PPG 신호 전체
            """
            # x = range(len(choice_num_1_use.loc['SpO2 Wave']))
            #
            # plt.plot(x, choice_num_1_use.loc['SpO2 Wave'])
            # # plt.savefig(plt_save_path + '\\{0}_3_Sample_Remove_Back.png'.format(num_1_file_name))
            # # plt.show()
            # plt.cla()

            choice_num_1_use = choice_num_1_use.transpose()
            first_interval = choice_num_1_use.iloc[spo2_wave_first[0]: spo2_wave_first[1], :]
            print(first_interval)
            # print(first_interval)

            first = first_interval.loc[:, "SpO2 Wave"].max()

            # print("in", first)

            first_index = first_interval.loc[:, "SpO2 Wave"]
            print("fi :",first_index)

            first_index = first_index.astype(float).idxmax()

            # print(first_index)

            spo2_wave_start.append(first_index)

            choice_num_1_use = choice_num_1_use.iloc[spo2_wave_start[i]:, :]

            # print("choice_num_1_use :", choice_num_1_use)

            """
            전체 PPG 신호의 평균값
            """
            spo2_wave_all = 0

            for spo2_wave in choice_num_1_use.loc[:, 'SpO2 Wave']:
                spo2_wave_all = spo2_wave + spo2_wave_all

            # print("spo2_wave_all :", spo2_wave_all)
            spo2_wave_all_avg = spo2_wave_all / len(choice_num_1_use.loc[:, 'SpO2 Wave'])
            # print("spo2_wave_all / len :", spo2_wave_all_avg)
            spo2_wave_all_avg_list.append(spo2_wave_all_avg)

            x = range(len(choice_num_1_use.loc[:, 'SpO2 Wave']))
            y = [spo2_wave_all_avg for _ in x]

            # plt.plot(x, y, c='darkorange')

            """
            -250 ~ 250 박스
            """
            # y = -250 + spo2_wave_all_avg
            # y = [y for _ in x]

            # plt.plot(x, y, c='k')

            # y = 250 + spo2_wave_all_avg
            # y = [y for _ in x]

            # plt.plot(x, y, c='k')
            # plt.savefig(plt_save_path + '\\{0}_6_wave_all_avg_line.png'.format(num_1_file_name))
            # plt.show()
            # plt.cla()

            x = choice_num_1_use.index

            # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
            # plt.show()
            # plt.cla()

            min = 1300
            min_1_count = 0
            j_over = 0

            for j in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
                if min > choice_num_1_use.iloc[j, 1]:
                    min = choice_num_1_use.iloc[j, 1]

                    if min_1_count > 0:
                        j_over = min_1_count
                        min_1_count = 0
                else:
                    min_1_count = min_1_count + 1
                    print("j", j)
                    if min_1_count == 20:
                        min_1 = choice_num_1_use.iloc[:j, 1].min()
                        min_1_idx = choice_num_1_use.iloc[:j, 1].astype(float).idxmin()

                        print("min_1 : ", min_1)
                        print("min_1_idx : ", min_1_idx)

                        min = 1300
                        min_1_count = 0
                        j = j - 20 + j_over
                        break

            max = -1200
            max_1_count = 0
            k_over = 0

            for k in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
                if max < choice_num_1_use.iloc[j + k, 1]:
                    max = choice_num_1_use.iloc[j + k, 1]

                    if max_1_count > 0:
                        k_over = max_1_count
                        max_1_count = 0
                else:
                    max_1_count = max_1_count + 1

                    if max_1_count == 10:
                        max_1 = choice_num_1_use.iloc[j: j + k, 1].max()
                        max_1_idx = choice_num_1_use.iloc[j: j + k, 1].astype(float).idxmax()

                        print("max_1 : ", max_1)
                        print("max_1_idx : ", max_1_idx)

                        max = -1200
                        max_1_count = 0
                        k = k - 10 + k_over
                        break

            for l in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
                if min > choice_num_1_use.iloc[j + k + l, 1]:
                    min = choice_num_1_use.iloc[j + k + l, 1]

                    if min_1_count > 0:
                        min_1_count = 0
                else:
                    min_1_count = min_1_count + 1

                    if min_1_count == 20:
                        min_2 = choice_num_1_use.iloc[j + k: j + k + l, 1].min()
                        min_2_idx = choice_num_1_use.iloc[j + k: j + k + l, 1].astype(float).idxmin()

                        print("min_2 : ", min_2)
                        print("min_2_idx : ", min_2_idx)

                        min_1_count = 0
                        break

            # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
            # plt.show()
            # plt.cla()

            """
            파형 한개의 크기를 표시
            빨간색 점 = 시작하는 지점
            파란색 점 = 파형의 최고점
            초록색 점 = 파형의 끝점
            """
            # # min 1
            # plt.scatter(min_1_idx, min_1, c = 'r')
            #
            # # max 1
            # plt.scatter(max_1_idx, max_1, c = 'b')
            #
            # # min 2
            # plt.scatter(min_2_idx, min_2, c = 'g')
            #
            # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
            # plt.savefig(plt_save_path + '\\{0}_4_OneWavePoint.png'.format(num_1_file_name))
            # plt.show()
            # plt.cla()

            # one_wave_len = range(min_1_idx, min_2_idx + 1)

            # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
            #
            # print(len(one_wave_len))
            # # print(len(choice_num_1_use.iloc[min_1_idx:min_2_idx, 1]))
            # print(len(choice_num_1_use.loc[min_1_idx:min_2_idx, 'SpO2 Wave']))
            #
            # plt.plot(one_wave_len, choice_num_1_use.loc[min_1_idx:min_2_idx, 'SpO2 Wave'], c = 'y')
            #
            # plt.show()
            # plt.cla()

            one_wave_len = min_2_idx - min_1_idx
            half_one_wave_len = one_wave_len // 2

            # print("one wave: ", one_wave_len)
            # print("half : ", half_one_wave_len)

            start_point = [choice_num_1_use.index[0]]
            end_point = [start_point[0] + 127]

            # print("start Point : ", start_point)
            # print("end Point : ", end_point)

            for m in end_point:
                if m + one_wave_len < choice_num_1_use.index[-1]:
                    check_point = m
                    check_point_end = check_point + one_wave_len

                    min_wave = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].min()
                    min_wave_idx = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].astype(float).idxmin()

                    check_point = min_wave_idx
                    check_point_end = check_point + (one_wave_len // 1.5)  # + (one_wave_len//2)
                    max_wave = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].max()
                    max_wave_idx = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].astype(float).idxmax()

                    start_point.append(max_wave_idx)
                    if max_wave_idx + 127 < choice_num_1_use.index[-1]:
                        end_point.append(max_wave_idx + 127)
                    else:
                        end_point.append(choice_num_1_use.index[-1])

            print("start Point : ", start_point)
            print("end Point : ", end_point)

            # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"], c='k')

            color = ["b", "g", "r", "c", "m", "y"]

            for n in range(len(start_point[:-1])):
                # print(len(start_point))

                x = range(start_point[n], end_point[n] + 1)
                y = choice_num_1_use.loc[start_point[n]:end_point[n], 'SpO2 Wave']

                # print(len(x))
                # print(len(y))

                # plt.plot(x, y, c=color[n%len(color)])

            # plt.savefig(plt_save_path + '\\{0}_5_128_Sampling.png'.format(num_1_file_name))
            # plt.show()
            # plt.cla()

            bio_index = []

            for p in range(128):
                bio_index.append(p)

            select_wave_one_avg_list = []

            end.append(start_point[-2])
            for o, wave_start in enumerate(start_point[:-1]):
                select_wave_one = copy.deepcopy(choice_num_1_use.loc[wave_start:end_point[o], 'SpO2 Wave'])

                # print(wave_start)
                # print(end_point[o])

                select_wave_value = []

                # print("Sele:", select_wave_one)
                # print("Sele_type:", type(select_wave_one))
                # print("Sele_len:", len(select_wave_one))

                num_tp = num_1_use_tp.transpose()

                print("num_tp :", num_tp)

                Max_BP_S_idx = num_tp.loc[:, ' BP_S'].astype(float).idxmax()
                Max_BP_D_idx = num_tp.loc[:, ' BP_D'].astype(float).idxmax()

                Max_BP_S = num_tp.loc[Max_BP_S_idx, ' BP_S']
                Max_BP_D = num_tp.loc[Max_BP_D_idx, ' BP_D']

                Max_BP_Series = pd.Series({'BP_S': Max_BP_S,
                                           'BP_D': Max_BP_D})

                # print(info_use)

                unique_number = info_use.iloc[i, 0]

                # print("height :", height)
                # print("weight :", weight)

                unique_number_Series = pd.Series({'unique_number': unique_number})

                select_wave_one_one = 0

                for r in select_wave_one:
                    select_wave_one_one = select_wave_one_one + r

                # print("select_wave_one_one : ", select_wave_one_one)

                # print("len :", len(select_wave_one))

                select_wave_one_avg = select_wave_one_one / len(select_wave_one)

                # print("select_wave_one_avg :", select_wave_one_avg)

                """
                값중에 1300이나 -1200이 들어가면 continue
                """
                # print("unique :", select_wave_one.unique())
                if 1300 in select_wave_one.unique():
                    continue
                if -1200 in select_wave_one.unique():
                    continue

                """
                부분 waveform 평균값이 전체 waveform 평균값 +- 250을 벗어난다면 continue
                """
                if ((select_wave_one_avg - spo2_wave_all_avg) > 250) or ((select_wave_one_avg - spo2_wave_all_avg) < -250):
                    continue

                select_wave_one_avg_list.append(select_wave_one_avg)

                x = range(wave_start - start_point[0], wave_start + 128 - start_point[0])
                y = [select_wave_one_avg for _ in x]

                # plt.plot(x, select_wave_one, c='cornflowerblue')  # cornflowerblue #steelblue
                # plt.plot(x, y, c=color[o % len(color)])

                """
                Waveform 값에서 전체 Waveform 평균 빼기
                """
                for s in range(len(select_wave_one)):
                    select_wave_one.iloc[s] = select_wave_one.iloc[s] - spo2_wave_all_avg

                print("SELE : ", select_wave_one)

                """
                1024로 나누는 정규화
                """
                for q in range(len(select_wave_one)):
                    select_wave_one.iloc[q] = select_wave_one.iloc[q] / 1024

                # plt.cla()
                # plt.plot(range(len(select_wave_one)), select_wave_one)
                # plt.axis([0, 127, -1, 1])
                # plt.show()
                # plt.savefig(plt_save_path + '\\{0}_11_{1}_each_waveform.png'.format(num_1_file_name, o))

                select_wave_one.index = bio_index

                select_wave_one = pd.concat([unique_number_Series, select_wave_one])

                # print(type(select_wave_one))
                # print("select_wave_oee:", select_wave_one)
                select_wave_pd = select_wave_pd.append(select_wave_one, ignore_index=True)
                # print("select_wave_pd:\n", select_wave_pd)

            select_wave_full_avg_list.append(select_wave_one_avg_list)

            # print("select_wave : \n", select_wave_pd)

            # plt.savefig(plt_save_path + '\\{0}_6_wave_all_avg_line.png'.format(num_1_file_name))
            # plt.savefig(plt_save_path + '\\{0}_7_wave_all_avg_box.png'.format(num_1_file_name))
            # plt.savefig(plt_save_path + '\\{0}_8_preprocessing.png'.format(num_1_file_name))
            # plt.savefig(plt_save_path + '\\{0}_9_final_sampling_full.png'.format(num_1_file_name))
            # plt.savefig(plt_save_path + '\\{0}_10_final_sampling_non_full.png'.format(num_1_file_name))
            # plt.show()
            # plt.cla()

        print("spo2_wave_all_avg_list :", spo2_wave_all_avg_list)
        print("select_wave_full_avg_list :", select_wave_full_avg_list)
        #
        print("start :", spo2_wave_start)
        print("end :", end)

        select_wave_pd.to_csv("collection.csv", mode='w', header=True)

    def learn(self, dir_name):
        """
        인공지능 모델 학습을 위한 함수
        :param dir_name:
            예측할 데이터가 있는 디렉터리명
            ex) data\\collection.csv
        :return:
        """
        path = self.set_path(dir_name)

        # print(path)

        csv_data = self.load_collection_data(path=path)
        # print(csv_data)

        ppg = csv_data.iloc[:,0:128]
        unique_num = csv_data.iloc[:,-1]

        print("ppg : ", ppg)
        print("unique_num : ", unique_num)

        """
        PPG 정보를 list로 변환
        """
        ppg_list = ppg.values.tolist()

        """
        키, 몸무게를 각 변수로 분리
        """
        unique_num = unique_num.values

        # 개인별 고유 번호를 2진수로 변환

        unique_num_binary_code_list = self.convert_DEC_to_BinaryCode(unique_num)

        print("Unique_num_binary_code_list : ", unique_num_binary_code_list)

        """
        
        """
        X_data = self.make_np_array(ppg_list)
        Y_data = self.make_np_array(unique_num_binary_code_list)

        # print("X_data:",X_data)
        # print("Y_data:",Y_data)

        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)#, random_state=seed)

        model, history = self.model(X_train, X_test, Y_train, Y_test)

        model.summary()
        print(model.evaluate(X_test, Y_test))

        print(history.history['val_loss'])
        print(history.history['loss'])

        # print(X_train[1:10])
        print(model.predict(X_train[1:2], batch_size=self.batch_size))
        # print(Y_train[1])

        predict = model.predict(X_test[:], batch_size=self.batch_size)
        Y_test_use = Y_test[:]

        print("X_test[1] : \n", X_test[0:10])
        print("model.predict(X_test[1:2], batch_size=5) : \n", predict)
        print("type(predict) : \n", type(predict))
        print("Y_test[1] : \n",Y_test_use)

        unique_num_dec, unique_num_Y_dec, per = self.postprocess(predict, Y_test_use)

        print("\n====Y 값====\n")
        print("BP_D_Y_dec", unique_num_Y_dec)

        print("\n====predict 값====\n")
        print("BP_D_dec", unique_num_dec)

        for i in range(len(unique_num_dec)):
            print("\n====Y 값====\n")
            print("BP_D_Y_dec", unique_num_Y_dec[i])

            print("\n====predict 값====\n")
            print("BP_D_dec", unique_num_dec[i])

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
        Y_dec_Series = pd.DataFrame([unique_num_Y_dec], index=['real_unique_num'])
        predict_dec_Serires = pd.DataFrame([unique_num_dec], index=['predict_unique_num'])
        per = pd.DataFrame([per], index=['gray code per'])

        Y_dec_Series = Y_dec_Series.transpose()
        predict_dec_Serires = predict_dec_Serires.transpose()
        per = per.transpose()

        save_csv = pd.concat([X_test_Series, Y_dec_Series, predict_dec_Serires, per], axis=1)

        print("save_csv : ", save_csv)

        self.save_prediction_data(save_csv, "prediction_learn.csv")

    def learn2(self, dir_name):
        """
        인공지능 모델 학습을 위한 함수
        :param dir_name:
            예측할 데이터가 있는 디렉터리명
            ex) data\\collection.csv
        :return:
        """
        path = self.set_path(dir_name)

        # print(path)

        csv_data = self.load_collection_data(path=path)
        # print(csv_data)

        wave = csv_data.iloc[:,0:128]
        BP = csv_data.iloc[:,-4:-2]
        HW = csv_data.iloc[:,-2:]

        # print(wave,"\n",BP, "\n", HW)

        """
        PPG 정보를 list로 변환
        """
        wave_list = wave.values.tolist()

        # print("wave_list", wave_list)
        # # print("wave_list", *wave_list, sep='\n')
        # print("wave_list[0]_len", len(wave_list[0]))
        # print("wave_list_len", len(wave_list))
        # print("wave_list_type", type(wave_list))

        """
        키, 몸무게를 각 변수로 분리
        """
        Height = HW.iloc[:, 0]
        Weight = HW.iloc[:, 1]

        Height = Height.values
        Weight = Weight.values

        # 키, 몸무게(십진수 데이터)를 gray code로 변환
        Height_gray_code_list = self.convert_DEC_to_GrayCode(Height)
        Weight_gray_code_list = self.convert_DEC_to_GrayCode(Weight)

        # print("Height_gray_code_list", Height_gray_code_list)
        # print("Weight_gray_code_list", Weight_gray_code_list)

        # 분리한 키, 몸무게의 gray code를 합침
        HW_gray_code_list = self.list_append(Height_gray_code_list, Weight_gray_code_list)

        # print("HW_gray_code_list", HW_gray_code_list)

        """
        혈압 정보를 수축기 혈압과 이완기 혈압으로 구분
        """
        BP_D = BP.iloc[:, 0]
        BP_S = BP.iloc[:, 1]

        BP_D = BP_D.values
        BP_S = BP_S.values

        # print(type(BP_D))

        BP_D_gray_code_list = self.convert_DEC_to_GrayCode(BP_D)
        BP_S_gray_code_list = self.convert_DEC_to_GrayCode(BP_S)

        # print("BP_D_gray_code_list", BP_D_gray_code_list)
        # print("BP_S_gray_code_list", BP_S_gray_code_list)

        X_np = self.list_append(wave_list, HW_gray_code_list)
        Y_np = self.list_append(BP_D_gray_code_list, BP_S_gray_code_list)

        X_data = self.make_np_array(X_np)
        Y_data = self.make_np_array(Y_np)

        # print("X_data:",X_data)
        # print("Y_data:",Y_data)

        model, history = self.model2(X_data, Y_data)

        model.summary()

        print("정확도 : ", model.evaluate(X_data, Y_data)[1])
        print("오차 : ", model.evaluate(X_data, Y_data)[0])

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

        model.add(tf.keras.layers.Dense(64, input_dim=128, activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(8, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=self.epoch, batch_size=self.batch_size, validation_data=(X_test, Y_test))

        self.save_model(model)

        # from keras.utils.vis_utils import plot_model
        # # os.environ["PATH"] += os.pathsep + 'C:\python\anaconda3_64\envs\Ai_2.0_20.12.31\Lib\site-packages\graphviz\bin'
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        #
        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')
        #
        # from keras import backend as K
        # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
        # print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        # print('Trainable params: {:,}'.format(trainable_count))
        # print('Non-trainable params: {:,}'.format(non_trainable_count))

        return model, history

    def model2(self, X_train, Y_train):
        """
        학습에 사용하는 인공지능 모델
        :param X_train:
        :param X_test:
        :param Y_train:
        :param Y_test:
        :return:
        """
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
        model.add(tf.keras.layers.Dense(16, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=self.epoch, batch_size=self.batch_size)

        self.save_model(model)

        # from keras.utils.vis_utils import plot_model
        # # os.environ["PATH"] += os.pathsep + 'C:\python\anaconda3_64\envs\Ai_2.0_20.12.31\Lib\site-packages\graphviz\bin'
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        #
        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')
        #
        # from keras import backend as K
        # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
        # print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        # print('Trainable params: {:,}'.format(trainable_count))
        # print('Non-trainable params: {:,}'.format(non_trainable_count))

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

        # print(path)

        csv_data = self.load_collection_data(path=path)
        # print(csv_data)

        wave = csv_data.iloc[:,0:128]
        BP = csv_data.iloc[:,-4:-2]
        HW = csv_data.iloc[:,-2:]

        # print(wave,"\n",BP, "\n", HW)

        wave_list = wave.values.tolist()

        # print("wave_list", wave_list)
        # # print("wave_list", *wave_list, sep='\n')
        # print("wave_list[0]_len", len(wave_list[0]))
        # print("wave_list_len", len(wave_list))
        # print("wave_list_type", type(wave_list))

        Height = HW.iloc[:, 0]
        Weight = HW.iloc[:, 1]

        Height = Height.values
        Weight = Weight.values

        Height_gray_code_list = self.convert_DEC_to_GrayCode(Height)
        Weight_gray_code_list = self.convert_DEC_to_GrayCode(Weight)

        # print("Height_gray_code_list", Height_gray_code_list)
        # print("Weight_gray_code_list", Weight_gray_code_list)

        HW_gray_code_list = self.list_append(Height_gray_code_list, Weight_gray_code_list)

        # print("HW_gray_code_list", HW_gray_code_list)

        BP_D = BP.iloc[:, 0]
        BP_S = BP.iloc[:, 1]

        BP_D = BP_D.values
        BP_S = BP_S.values

        # print(type(BP_D))

        BP_D_gray_code_list = self.convert_DEC_to_GrayCode(BP_D)
        BP_S_gray_code_list = self.convert_DEC_to_GrayCode(BP_S)

        # print("BP_D_gray_code_list", BP_D_gray_code_list)
        # print("BP_S_gray_code_list", BP_S_gray_code_list)

        X_np = self.list_append(wave_list, HW_gray_code_list)
        Y_np = self.list_append(BP_D_gray_code_list, BP_S_gray_code_list)

        X_data = self.make_np_array(X_np)
        Y_data = self.make_np_array(Y_np)

        # print("X_data:",X_data)
        # print("Y_data:",Y_data)

        model = self.load_model()

        model.summary()
        print(model.evaluate(X_data, Y_data))

        # print(X_train[1:10])
        print(model.predict(X_data[1:2], batch_size=self.batch_size))
        print(Y_data[1])

        predict = model.predict(X_data[:], batch_size=self.batch_size)
        Y_test_use = Y_data[:]

        print("X_test[1] : \n", X_data[0:10])
        print("model.predict(X_test[1:2], batch_size=5) : \n", predict)
        # print("type(predict) : \n", type(predict))
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

        print("정확도 : ", model.evaluate(X_data, Y_data)[1])
        print("오차 : ", model.evaluate(X_data, Y_data)[0])

        X_test_Series = pd.DataFrame(X_data)
        Y_test_Series = pd.DataFrame(Y_data)
        Y_dec_Series = pd.DataFrame([BP_D_Y_dec, BP_S_Y_dec], index=['real_BP_D', 'real_BP_S'])
        predict_dec_Serires = pd.DataFrame([BP_D_dec, BP_S_dec], index=['predict_BP_D', 'predict_BP_S'])
        per = pd.DataFrame([per], index=['gray code per'])

        Y_dec_Series = Y_dec_Series.transpose()
        predict_dec_Serires = predict_dec_Serires.transpose()
        per = per.transpose()

        save_csv = pd.concat([X_test_Series, Y_dec_Series, predict_dec_Serires, per], axis=1)

        print("save_csv : ", save_csv)

        self.save_prediction_data(save_csv, "prediction_predict.csv")

    def postprocess(self, predict, Y_test_use):
        """
        후처리 과정
        :param predict:
            인공지능 모델링을 통해 얻은 순수 예측값
        :param Y_test_use:
            실측한 데이터의 Y값(수축기 혈압, 이완기 혈압)
        :return:
            predction(수축기 혈압, 이완기 혈압), 실측 값(수축기 혈압, 이완기 혈압), 예측값과 실측값 사이의 정확도(%)
        """
        predict_list = self.convert_NP_to_LIST(predict)
        Y_test_use_list = self.convert_NP_to_LIST(Y_test_use)

        gray_list = self.convert_Prediction_to_GrayCode(predict_list)
        # Y_test_use_gray_list = self.convert_Prediction_to_GrayCode(Y_test_use_list)

        per = self.match(Y_test_use_list, gray_list)
        # print("predict_list :", predict_list)
        # print("gray_list :\n", *gray_list, sep='\n')

        print("matching per : ", per)

        unique_num_dec = self.convert_BinaryCode_to_DEC(gray_list)
        unique_num_Y_dec = self.convert_BinaryCode_to_DEC(Y_test_use_list)

        return unique_num_dec, unique_num_Y_dec, per

    def search_csv_file(self, file_list):
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

    def set_path(self, dir_name):
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
        model = load_model('data\\model_save.h5')

        return model

    def save_prediction_data(self, save, file_name):
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

    def load_collection_data(self, path):
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

    def match(self, real_data, predict_data):
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

    def make_np_array(self, data):
        """
        리스트 데이터를 텐서플로우에서 사용할 수 있게 numpy의 array로 변경
        :param data:
            리스트 자료형 데이터
        :return:
            array 자료형 데이터
        """
        result = np.array(data)

        return result

    def list_append(self, BP_D, BP_S):
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

    def convert_NP_to_LIST(self, np, dtype=None):
        """
        numpy의 array 형식의 데이터를 list로 변환
        :param np:
        :param dtype:
        :return:
            list 자료형의 값
        """
        list = np.tolist()

        return list

    def convert_Prediction_to_GrayCode(self, prediction_list):
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
            half = 0.5

            predict_to_binary = []

            for element in predict:
                if element > half:
                    bin = 1
                elif element < half:
                    bin = 0
                predict_to_binary.append(bin)
            binary_list.append(predict_to_binary)

        return binary_list

    """
    Gray code를 십진수로 Decoding하는 부분
    """
    def convert_GrayCode_to_DEC(self, gray_code_list):
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

    def convert_GrayCode_to_BinaryCode(self, gray):
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

    def convert_BinaryCode_to_DEC(self, binary):
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
    def convert_DEC_to_GrayCode(self, dec):
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

    def convert_BinaryCode_to_GrayCode(self, binary):
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

    def convert_DEC_to_BinaryCode(self, dec):
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

    # def XOR(self, num1, num2):
    #     num1

if __name__ == "__main__":
    A = DistinguishingPeopleByPpgSignals()

    # A.rp_preprocess()
    # A.monitoring_preprocess("data/Collection", "data/info.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data\\unknown.csv")

    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data\\unknown.csv")

    # A.predict("data/collection_new.csv")

    # A.monitoring_preprocess("data/Collection", "data/info.csv")
    # A.learn("data\\collection.csv")
    # A.predict("data/collection.csv")

    # A.monitoring_preprocess("data/Collection/new_100", "data/info_100.csv")

    # 100개 데이터 전부 학습
    # 정확도 : 0.8297
    # 오차   : 0.3303

    # 28개 데이터 전부 학습
    # 정확도 : 0.9597
    # 오차   : 0.1087

    # A.learn("data/100/collection.csv")
    # A.predict("data/collection.csv")

    # A.monitoring_preprocess("data", "info.csv")
    A.learn("data_set_ppg&unique_num.csv")
