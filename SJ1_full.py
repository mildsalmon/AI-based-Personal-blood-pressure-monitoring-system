# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import pandas as pd

# def __init__(self):
#     self.wave_start_point
#     self.wave_end_point
#
#     pd.set_option('display.max_columns', 100)
#     pd.set_option('display.width', 1000)
#
#     self.sj1_csv_people = {
#         # people 1
#         1: self.select_people(0),
#         # people 2
#         2: self.select_people(1),
#         # people 3
#         3: self.select_people(2),
#         # people 4
#         4: self.select_people(3),
#         # people 5
#         5: self.select_people(4)
#     }

def select_people(num):
    sj1_csv = pd.read_csv('data/SJ1.csv', engine='python')

    print(sj1_csv)
    #
    # print(sj1_csv.shape)
    #
    # print(sj1_csv.dropna(axis=0, how='any'))

    sj1_csv_exam = sj1_csv.dropna(axis=0, how='any')
    sj1_csv_people = sj1_csv_exam.iloc[num, :]

    return sj1_csv_people

def setting(sj1_csv_people_select, people_num):
    if people_num == 1:
        wave_start_point = 1289
        wave_end_point = 6505
    elif people_num == 2:
        wave_start_point = 5
        wave_end_point = 0
    elif people_num == 3:
        wave_start_point = 1289
        wave_end_point = 6505
    elif people_num == 4:
        wave_start_point = 1289
        wave_end_point = 6505
    elif people_num == 5:
        wave_start_point = 1289
        wave_end_point = 6505
    #
    # self.wave_start_point = wave_start_point
    # self.wave_end_point = wave_end_point


# f = open('data/SJ1.csv', 'r')#, encoding='utf-8')
# reader = csv.reader(f)

# for line in reader:
#     print(len(line))
#     print(line)
#
# f.close()
#
# plt.plot(line)
# plt.show()

if __name__ == "__main__":

    # 판다스 크기 키우기
    # pd.set_option('display.max_rows', 500)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)

    sj1_csv_people = {
        # people 1
        1: select_people(0),
        # people 2
        2: select_people(1),
        # people 3
        3: select_people(2),
        # people 4
        4: select_people(3),
        # people 5
        5: select_people(4)
    }

    people_num = 5

    sj1_csv_people_select = sj1_csv_people[people_num]

    # people 1
    # sj1_csv_people1 = sj1_csv_exam.iloc[0, :]

    # people 2
    # sj1_csv_people1 = sj1_csv_exam.iloc[1, :]

    # people 3
    # sj1_csv_people1 = sj1_csv_exam.iloc[2, :]

    # people 4
    # sj1_csv_people1 = sj1_csv_exam.iloc[3, :]

    # people 5
    # sj1_csv_people1 = sj1_csv_exam.iloc[4, :]

    print(sj1_csv_people_select)


    biosignal = sj1_csv_people_select[1:5]
    # biowave = sj1_csv_people1[5:6675]

    if people_num == 1:
        # people1의 wave 30개 선정
        biowave = sj1_csv_people_select[1289:6505]
        x = range(len(biowave))

        print(biosignal)
        print(biowave)

        print(x)

        # plt.rcParams["figure.figsize"] = (1000,1000)
        # plt.rcParams["figure.figsize"] = (25, 10)
        # plt.plot(x, biowave)
        # plt.show()

        min = biowave.astype(float).min()
        # max = biowave.astype(float).max()
        print("min", min)
        # print("max", max)

        ############

        # for i in range(len(biowave)):
        #     biowave.iloc[i] = biowave.iloc[i] - min
        #
        # x = range(len(biowave))
        #
        # plt.plot(x, biowave)
        # plt.show()
        #
        # min = biowave.astype(float).min()
        # max = biowave.astype(float).max()
        # print("min", min)
        # print("max", max)
        #
        # ###########

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min
            # print(i, " ", biowave.iloc[i])

        max = biowave.astype(float).max()
        print("max", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max
            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(x, biowave)
        plt.axis([0, len(biowave), 0, 1])
        plt.show()

        min = biowave.astype(float).min()
        max = biowave.astype(float).max()
        print("min", min)
        print("max", max)

    elif people_num == 2:
        # people2의 wave 30개 선정
        biowave = sj1_csv_people_select[1:]
        x = range(len(biowave))

        print(biosignal)
        print(biowave)

        print(x)

        # plt.rcParams["figure.figsize"] = (25, 10)
        # plt.plot(x, biowave)

        min = biowave.astype(float).min()
        print("min", min)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min

        max = biowave.astype(float).max()
        print("max", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max
            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(x, biowave)
        plt.axis([0, len(biowave), 0, 1])
        plt.show()

        min = biowave.astype(float).min()
        max = biowave.astype(float).max()
        print("min", min)
        print("max", max)

    elif people_num == 3:
        # people2의 wave 30개 선정
        biowave = sj1_csv_people_select[5:]
        x = range(len(biowave))

        print(biosignal)
        print(biowave)

        print(x)

        # plt.rcParams["figure.figsize"] = (25, 10)
        # plt.plot(x, biowave)

        min = biowave.astype(float).min()
        print("min", min)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min

        max = biowave.astype(float).max()
        print("max", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max
            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(x, biowave)
        plt.axis([0, len(biowave), 0, 1])
        plt.show()

        min = biowave.astype(float).min()
        max = biowave.astype(float).max()
        print("min", min)
        print("max", max)

    elif people_num == 4:
        # people2의 wave 30개 선정
        biowave = sj1_csv_people_select[5:]
        x = range(len(biowave))

        print(biosignal)
        print(biowave)

        print(x)

        # plt.rcParams["figure.figsize"] = (25, 10)
        # plt.plot(x, biowave)

        min = biowave.astype(float).min()
        print("min", min)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min

        max = biowave.astype(float).max()
        print("max", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max
            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(x, biowave)
        plt.axis([0, len(biowave), 0, 1])
        plt.show()

        min = biowave.astype(float).min()
        max = biowave.astype(float).max()
        print("min", min)
        print("max", max)


    elif people_num == 5:
        # people2의 wave 30개 선정
        biowave = sj1_csv_people_select[5:]
        x = range(len(biowave))

        print(biosignal)
        print(biowave)

        print(x)

        # plt.rcParams["figure.figsize"] = (25, 10)
        # plt.plot(x, biowave)

        min = biowave.astype(float).min()
        print("min", min)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min

        max = biowave.astype(float).max()
        print("max", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max
            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(x, biowave)
        plt.axis([0, len(biowave), 0, 1])
        plt.show()

        min = biowave.astype(float).min()
        max = biowave.astype(float).max()
        print("min", min)
        print("max", max)