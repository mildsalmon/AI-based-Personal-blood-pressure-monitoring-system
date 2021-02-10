# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

read_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\수집")
# read_path = "D:\CloudStation\SourceCode\MAI\data\수집"
print(read_path)

read_path_list = os.listdir(read_path)[1:]
# read_path_list = ["SpO2_20210118163842.csv"]

print(read_path_list)

spo2_wave_first = [
    [850, 882],
    [269, 282],
    [15, 50],
    [176, 217],
    [705, 745],
    [11, 46],
    [31, 50],
    [96, 125],
    [13, 35],
    [56, 90],
    [39, 52],
    [6, 22],
    [197, 220],
    [12, 47],
    [425, 440],
    [417, 430],
    [697, 731],
    [36, 68],
    [1068, 1084],
    [2156, 2194],
    [678, 712],
    [31, 47],
    [40, 64],
    [32, 69],
    [19, 35],
]

spo2_wave_start = []

select_wave = []
select_wave_pd = pd.DataFrame()


for i, path in enumerate(read_path_list):
    num_1_file_name = path

    print(num_1_file_name)

    num_1_path = read_path + "\\" + num_1_file_name

    num_1_pd = pd.read_csv(num_1_path)

    print(num_1_pd)

    # num_1_pd_tp = num_1_pd.transpose()

    # print(num_1_pd_tp)

    # num_1_pd_tp_info = num_1_pd_tp.info()
    num_1_pd_info = num_1_pd.info()

    # print(num_1_pd_tp_info)
    print(num_1_pd_info)
    print(num_1_pd.index)

    # print(num_1_pd['SpO2'])

    print(type(num_1_pd[['SpO2 Wave', ' SpO2', ' BP_S', ' BP_D', ' TIME']]))
    num_1_use = num_1_pd[[' SpO2', 'SpO2 Wave', ' BP_S', ' BP_D', ' TIME']]

    print(num_1_use)

    num_1_use_tp = num_1_use.transpose()

    print(num_1_use_tp)

    print(num_1_use_tp.loc['SpO2 Wave'].min())
    print(num_1_use_tp.loc['SpO2 Wave'].max())

    x = range(len(num_1_use_tp.loc['SpO2 Wave']))
    print(len(num_1_use_tp.loc['SpO2 Wave']))

    plt_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image\\collection")
    print(plt_save_path)

    plt.rcParams["figure.figsize"] = (18, 7)
    plt.plot(x, num_1_use_tp.loc['SpO2 Wave'])
    plt.axis([0, len(num_1_use_tp.loc['SpO2 Wave']), num_1_use_tp.loc['SpO2 Wave'].min(), num_1_use_tp.loc['SpO2 Wave'].max()])
    # plt.savefig(plt_save_path + "\\{0}_Full.png".format(num_1_file_name))
    # plt.show()
    plt.cla()

    x1 = range(4601)
    x2 = range(4600, len(num_1_use_tp.loc['SpO2 Wave'])-1)

    plt.plot(x1, num_1_use_tp.iloc[1, 0:4601])
    plt.plot(x2, num_1_use_tp.iloc[1, 4600:len((num_1_use_tp.loc['SpO2 Wave']))-1], 'r')
    plt.axis([0, len(num_1_use_tp.loc['SpO2 Wave']), num_1_use_tp.loc['SpO2 Wave'].min(), num_1_use_tp.loc['SpO2 Wave'].max()])
    # plt.savefig(plt_save_path + '\\{0}_Full_Red.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

    choice_num_1_use = num_1_use_tp.iloc[:,:4600]

    print("cho", choice_num_1_use)

    x = range(len(choice_num_1_use.loc['SpO2 Wave']))

    plt.plot(x, choice_num_1_use.loc['SpO2 Wave'])
    # plt.savefig(plt_save_path + '\\{0}_Sample_Remove_Back.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

    choice_num_1_use = choice_num_1_use.transpose()
    # for i in range(4600):
    first_interval = choice_num_1_use.iloc[spo2_wave_first[i][0]: spo2_wave_first[i][1],:]

    print(first_interval)

    first = first_interval.loc[:,"SpO2 Wave"].max()

    print("in", first)

    first_index = first_interval.loc[:,"SpO2 Wave"]

    first_index = first_index.astype(float).idxmax()

    print(first_index)

    spo2_wave_start.append(first_index)
    # real_pd = choice_num_1_use[]

    choice_num_1_use = choice_num_1_use.iloc[spo2_wave_start[i]:,:]

    print(choice_num_1_use)

    # x = range(len(choice_num_1_use.loc[:,'SpO2 Wave']))
    x = choice_num_1_use.index

    # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
    # plt.show()
    # plt.cla()

    min = 1300
    min_1_count = 0
    j_over = 0

    for j in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
        # print("choiloc", choice_num_1_use.iloc[j, 1])

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

        # print(min)

    # print("j",j)
    # print("j_over",j_over)
    max = -1200
    max_1_count = 0
    k_over = 0

    for k in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
        if max < choice_num_1_use.iloc[j + k, 1]:
            max = choice_num_1_use.iloc[j + k, 1]
            # print(choice_num_1_use.iloc[j + k, 1])
            # print("jk",j+k)

            if max_1_count > 0:
                k_over = max_1_count
                max_1_count = 0

        else:
            max_1_count = max_1_count + 1

            if max_1_count == 10:
                max_1 = choice_num_1_use.iloc[j : j + k, 1].max()
                max_1_idx = choice_num_1_use.iloc[j : j + k, 1].astype(float).idxmax()

                print("max_1 : ", max_1)
                print("max_1_idx : ", max_1_idx)

                max = -1200
                max_1_count = 0
                k = k - 10 + k_over
                break
        # print(max)


    for l in range(len(choice_num_1_use.loc[:, 'SpO2 Wave'])):
        # print("choiloc", choice_num_1_use.iloc[j, 1])
        # print(min_1_count)
        # print(min)
        if min > choice_num_1_use.iloc[j + k + l, 1]:
            min = choice_num_1_use.iloc[j + k + l, 1]

            if min_1_count > 0:
                min_1_count = 0

        else:
            min_1_count = min_1_count + 1

            if min_1_count == 20:
                min_2 = choice_num_1_use.iloc[j + k : j + k + l, 1].min()
                min_2_idx = choice_num_1_use.iloc[j + k : j + k + l, 1].astype(float).idxmin()

                print("min_2 : ", min_2)
                print("min_2_idx : ", min_2_idx)

                min_1_count = 0
                break

        # print(min)

    # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
    # plt.show()
    # plt.cla()

    # # min 1
    # plt.scatter(min_1_idx, min_1, c = 'r')
    #
    # # max 1
    # plt.scatter(max_1_idx, max_1, c = 'b')
    #
    # # min 2
    # plt.scatter(min_2_idx, min_2, c = 'g')

    # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
    # plt.show()
    # plt.cla()

    one_wave_len = range(min_1_idx, min_2_idx+1)

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

    print("one wave: ",one_wave_len)
    print("half : ", half_one_wave_len)

    start_point = [choice_num_1_use.index[0]]
    end_point = [start_point[0]+128]

    print("start Point : ", start_point)
    print("end Point : ", end_point)

    # 설계 실수, 1 wave 길이를 집어넣어야함
    # for m in end_point:
    #     if m+half_one_wave_len < choice_num_1_use.index[-1]:
    #         max_wave = choice_num_1_use.loc[m:m+half_one_wave_len, 'SpO2 Wave'].max()
    #         max_wave_idx = choice_num_1_use.loc[m:m+half_one_wave_len, 'SpO2 Wave'].astype(float).idxmax()
    #
    #         start_point.append(max_wave_idx)
    #         if max_wave_idx + 128 < choice_num_1_use.index[-1]:
    #             end_point.append(max_wave_idx+128)
    #         else:
    #             end_point.append(choice_num_1_use.index[-1])

    for m in end_point:
        if m+one_wave_len < choice_num_1_use.index[-1]:
            max_wave = choice_num_1_use.loc[m:m+one_wave_len, 'SpO2 Wave'].max()
            max_wave_idx = choice_num_1_use.loc[m:m+one_wave_len, 'SpO2 Wave'].astype(float).idxmax()

            start_point.append(max_wave_idx)
            if max_wave_idx + 128 < choice_num_1_use.index[-1]:
                end_point.append(max_wave_idx+128)
            else:
                end_point.append(choice_num_1_use.index[-1])

    print("start Point : ", start_point)
    print("end Point : ", end_point)

    plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"], c='k')

    color = ["b", "g", "r", "c", "m", "y"]

    for n in range(len(start_point)):
        print(len(start_point))

        x = range(start_point[n], end_point[n]+1)
        y = choice_num_1_use.loc[start_point[n]:end_point[n], 'SpO2 Wave']

        print(len(x))
        print(len(y))

        plt.plot(x, y, c=color[n%len(color)])

    # plt.show()
    plt.cla()

    bio_index = []

    for i in range(129):
        bio_index.append(i)

    for o, wave_start in enumerate(start_point[:-1]):
        select_wave_one = choice_num_1_use.loc[wave_start:end_point[o], 'SpO2 Wave']

        select_wave_value = []

        # print(len(select_wave_one))
        # print(len(select_wave_one)-1)
        for i in range(len(select_wave_one)):
            select_wave_one.iloc[i] = select_wave_one.iloc[i]/1300

        select_wave_one.index = bio_index

        print("select_wave_oee:",select_wave_one)
        select_wave_pd = select_wave_pd.append(select_wave_one, ignore_index=True)
        print("select_wave_pd:",select_wave_pd)
        # for v in select_wave_one:
        #     print("value: ", v)
        #     print(v/1300)
        #     # 데이터 정규화가 잘 되었는지 확인
        #     # if v == 1300:
        #         # input()
        #     select_wave_value.append(v/1300)
        #
        # select_wave_value_Series = pd.Series(select_wave_value)
        # print("Series: ",select_wave_value_Series)
        # # select_wave.append(select_wave_value)
        # # select_wave_pd.append(select_wave, ignore_index=True)
        # select_wave_pd.append(select_wave_value_Series.to_frame())
        # # 리스트를 줄바꿈하여 출력하려면 * (unpacking operator)를 사용하고 sep으로 개행한다.
        # # print(*select_wave, sep="\n")

    # print("select_wave : ", *select_wave, sep="\n")
    print("select_wave : ", select_wave_pd)

print(spo2_wave_start)

