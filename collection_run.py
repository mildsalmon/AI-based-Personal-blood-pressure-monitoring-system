# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import copy

pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

read_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\수집")
# read_path = "D:\CloudStation\SourceCode\MAI\data\수집"
print(read_path)

read_path_list = os.listdir(read_path)[2:]
# read_path_list = ["SpO2_20210118163842.csv"]

print(read_path_list)

"""
spo2 시작할때 발생하는 이상치를 제거하는 작업
학습에 사용할 데이터의 시작지점을 대략적으로 표시한 것
수동 작업 -> 추후 자동화 예정
"""
# spo2_wave_first = [
#     [850, 882],
#     [269, 282],
#     [15, 50],
#     [176, 217],
#     [705, 745],
#     [11, 46],
#     [31, 50],
#     [96, 125],
#     [13, 35],
#     [56, 90],
#     [39, 52],
#     [6, 22],
#     [197, 220],
#     [12, 47],
#     [425, 440],
#     [417, 430],
#     [697, 731],
#     [36, 68],
#     [1068, 1084],
#     [2156, 2194],
#     [678, 712],
#     [31, 47],
#     [40, 64],
#     [32, 69],
#     [19, 35],
# ]
spo2_wave_first = [0, 60]

"""
1개의 waveform에서 학습을 위해 시작하는 지점(1개의 waveform에서 최고점)을 선택해서 spo2_wave_start 리스트에 추가함
"""
spo2_wave_start = []

# select_wave = []
"""
128개로 sampling한 waveform, 키, 몸무게를 한 묶음으로 데이터프레임에 추가함
"""
select_wave_pd = pd.DataFrame()

spo2_wave_all_avg_list = []
select_wave_full_avg_list = []

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
    num_1_use = num_1_pd[[' SpO2', 'SpO2 Wave', ' BP_S', ' BP_D', ' TIME', 'height', 'weight']]

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
    # plt.savefig(plt_save_path + "\\{0}_1_Full.png".format(num_1_file_name))
    # plt.show()
    plt.cla()

    x1 = range(4601)
    x2 = range(4600, len(num_1_use_tp.loc['SpO2 Wave'])-1)

    plt.plot(x1, num_1_use_tp.iloc[1, 0:4601])
    plt.plot(x2, num_1_use_tp.iloc[1, 4600:len((num_1_use_tp.loc['SpO2 Wave']))-1], 'r')
    plt.axis([0, len(num_1_use_tp.loc['SpO2 Wave']), num_1_use_tp.loc['SpO2 Wave'].min(), num_1_use_tp.loc['SpO2 Wave'].max()])
    # plt.savefig(plt_save_path + '\\{0}_2_Full_Red.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

    choice_num_1_use = num_1_use_tp.iloc[:,:4600]

    print("cho", choice_num_1_use)

    x = range(len(choice_num_1_use.loc['SpO2 Wave']))

    plt.plot(x, choice_num_1_use.loc['SpO2 Wave'])
    # plt.savefig(plt_save_path + '\\{0}_3_Sample_Remove_Back.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

    choice_num_1_use = choice_num_1_use.transpose()
    # for i in range(4600):
    first_interval = choice_num_1_use.iloc[spo2_wave_first[0]: spo2_wave_first[1],:]

    print(first_interval)

    first = first_interval.loc[:,"SpO2 Wave"].max()

    print("in", first)

    first_index = first_interval.loc[:,"SpO2 Wave"]

    first_index = first_index.astype(float).idxmax()

    print(first_index)

    spo2_wave_start.append(first_index)
    # real_pd = choice_num_1_use[]

    choice_num_1_use = choice_num_1_use.iloc[spo2_wave_start[i]:,:]

    print("choice_num_1_use :", choice_num_1_use)

    spo2_wave_all = 0

    for spo2_wave in choice_num_1_use.loc[:, 'SpO2 Wave']:
        spo2_wave_all = spo2_wave + spo2_wave_all

    print("spo2_wave_all :", spo2_wave_all)
    spo2_wave_all_avg = spo2_wave_all / len(choice_num_1_use.loc[:, 'SpO2 Wave'])
    print("spo2_wave_all / len :", spo2_wave_all_avg)
    spo2_wave_all_avg_list.append(spo2_wave_all_avg)

    x = range(len(choice_num_1_use.loc[:, 'SpO2 Wave']))
    y = [spo2_wave_all_avg for _ in x]

    # plt.plot(x, choice_num_1_use.loc[:,'SpO2 Wave'], c='lawngreen')

    plt.plot(x, y, c='darkorange')

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
    #
    plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"])
    # plt.savefig(plt_save_path + '\\{0}_4_OneWavePoint.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

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
    end_point = [start_point[0]+127]

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
            check_point = m
            check_point_end = check_point + one_wave_len

            min_wave = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].min()
            min_wave_idx = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].astype(float).idxmin()

            # check_point = m + (one_wave_len//3)
            check_point = min_wave_idx
            check_point_end = check_point + (one_wave_len//1.5) # + (one_wave_len//2)
            max_wave = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].max()
            max_wave_idx = choice_num_1_use.loc[check_point:check_point_end, 'SpO2 Wave'].astype(float).idxmax()

            start_point.append(max_wave_idx)
            if max_wave_idx + 127 < choice_num_1_use.index[-1]:
                end_point.append(max_wave_idx+127)
            else:
                end_point.append(choice_num_1_use.index[-1])
        # print("c_index:", choice_num_1_use.index[-1])
        # print(m)

    print("start Point : ", start_point)
    print("end Point : ", end_point)

    # plt.plot(x, choice_num_1_use.loc[:, "SpO2 Wave"], c='k')

    color = ["b", "g", "r", "c", "m", "y"]

    for n in range(len(start_point[:-1])):
        print(len(start_point))

        x = range(start_point[n], end_point[n]+1)
        y = choice_num_1_use.loc[start_point[n]:end_point[n], 'SpO2 Wave']

        print(len(x))
        print(len(y))

        # plt.plot(x, y, c=color[n%len(color)])

    # plt.savefig(plt_save_path + '\\{0}_5_128_Sampling.png'.format(num_1_file_name))
    # plt.show()
    # plt.cla()

    bio_index = []

    for p in range(128):
        bio_index.append(p)

    select_wave_one_avg_list = []

    for o, wave_start in enumerate(start_point[:-1]):
        # select_wave_one = pd.Series()
        # select_wave_one = (choice_num_1_use.loc[wave_start:end_point[o], 'SpO2 Wave'])
        select_wave_one = copy.deepcopy(choice_num_1_use.loc[wave_start:end_point[o], 'SpO2 Wave'])

        print(wave_start)
        print(end_point[o])

        select_wave_value = []

        print("Sele:",select_wave_one)
        print("Sele_type:",type(select_wave_one))
        print("Sele_len:", len(select_wave_one))

        num_tp = num_1_use_tp.transpose()

        print("num_tp :", num_tp)

        Max_BP_S_idx = num_tp.loc[:,' BP_S'].astype(float).idxmax()
        # Min_BP_S_pd = choice_num_1_use.loc[:,'BP_S'].min()
        Max_BP_D_idx = num_tp.loc[:,' BP_D'].astype(float).idxmax()
        # Min_BP_D_pd = choice_num_1_use.loc[:,'BP_D'].min()

        Max_BP_S = num_tp.loc[Max_BP_S_idx, ' BP_S']
        Max_BP_D = num_tp.loc[Max_BP_D_idx, ' BP_D']

        Max_BP_Series = pd.Series({'BP_S':Max_BP_S,
                                     'BP_D':Max_BP_D})

        height = num_tp.loc[0, "height"]
        weight = num_tp.loc[0, "weight"]

        h_w_Series = pd.Series({'height':height,
                                'weight':weight})

        # print("BP:",Max_BP_S_pd, Max_BP_D_pd)
        # print("BP:",type(Max_BP_S_pd), type(Max_BP_D_pd))

        # BP_data = pd.DataFrame([Max_BP_S_pd, Max_BP_D_pd])

        # BP_data = pd.DataFrame(Max_BP_Series)
        # BP_data = BP_data.transpose()

        # print("BP_Data:\n",BP_data)
        # print(len(select_wave_one))
        # print(len(select_wave_one)-1)

        select_wave_one_one = 0

        for r in select_wave_one:
            select_wave_one_one = select_wave_one_one + r

        print("select_wave_one_one : ", select_wave_one_one)

        print("len :", len(select_wave_one))

        select_wave_one_avg = select_wave_one_one / len(select_wave_one)

        print("select_wave_one_avg :", select_wave_one_avg)

        """
        값중에 1300이나 -1200이 들어가면 continue
        """
        print("unique :", select_wave_one.unique())
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

        x = range(wave_start-start_point[0], wave_start+128-start_point[0])
        y = [select_wave_one_avg for _ in x]

        plt.plot(x, select_wave_one, c='cornflowerblue') #cornflowerblue #steelblue
        plt.plot(x, y, c=color[o%len(color)])

        """
        Waveform 값에서 전체 Waveform 평균 빼기
        """
        for s in range(len(select_wave_one)):
            select_wave_one.iloc[s] = select_wave_one.iloc[s] - spo2_wave_all_avg

        print("SELE : ", select_wave_one)

        """
        이전에 하던 정규화
        """
        # for q in range(len(select_wave_one)):
        #     if select_wave_one.iloc[q] > 0:
        #         select_wave_one.iloc[q] = select_wave_one.iloc[q]/1300
        #     elif select_wave_one.iloc[q] < 0:
        #         select_wave_one.iloc[q] = select_wave_one.iloc[q]/1200
        #     else:
        #         select_wave_one.iloc[q] = 0

        """
        1024로 나누는 정규화
        """
        for q in range(len(select_wave_one)):
            select_wave_one.iloc[q] = select_wave_one.iloc[q]/1024

        # plt.cla()
        # plt.plot(range(len(select_wave_one)), select_wave_one)
        # plt.axis([0, 127, -1, 1])
        # plt.show()
        # plt.savefig(plt_save_path + '\\{0}_11_{1}_each_waveform.png'.format(num_1_file_name, o))

        select_wave_one.index = bio_index

        select_wave_one = pd.concat([h_w_Series, Max_BP_Series, select_wave_one])

        print(type(select_wave_one))
        print("select_wave_oee:",select_wave_one)
        select_wave_pd = select_wave_pd.append(select_wave_one, ignore_index=True)
        print("select_wave_pd:\n",select_wave_pd)
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
    select_wave_full_avg_list.append(select_wave_one_avg_list)

    print("select_wave : \n", select_wave_pd)

    # plt.savefig(plt_save_path + '\\{0}_6_wave_all_avg_line.png'.format(num_1_file_name))
    # plt.savefig(plt_save_path + '\\{0}_7_wave_all_avg_box.png'.format(num_1_file_name))
    # plt.savefig(plt_save_path + '\\{0}_8_preprocessing.png'.format(num_1_file_name))
    # plt.savefig(plt_save_path + '\\{0}_9_final_sampling_full.png'.format(num_1_file_name))
    # plt.savefig(plt_save_path + '\\{0}_10_final_sampling_non_full.png'.format(num_1_file_name))
    # plt.show()
    plt.cla()

print("spo2_wave_all_avg_list :", spo2_wave_all_avg_list)
print("select_wave_full_avg_list :", select_wave_full_avg_list)

print(spo2_wave_start)



select_wave_pd.to_csv("data/collection.csv", mode='w', header=True)
