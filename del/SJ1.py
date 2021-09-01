# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import pandas as pd

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


# 판다스 크기 키우기
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

sj1_csv = pd.read_csv('data/SJ1.csv', engine='python')


print(sj1_csv)

print(sj1_csv.shape)

print(sj1_csv.dropna(axis=0, how='any'))

sj1_csv_exam = sj1_csv.dropna(axis=0, how='any')

sj1_csv_people1 = sj1_csv_exam.iloc[0, :]
sj1_csv_people2 = sj1_csv_exam.iloc[1, :]
sj1_csv_people3 = sj1_csv_exam.iloc[2, :]
sj1_csv_people4 = sj1_csv_exam.iloc[3, :]
sj1_csv_people5 = sj1_csv_exam.iloc[4, :]

print(sj1_csv_people1)


# biosignal = sj1_csv_people1[1:5]
# biowave = sj1_csv_people1[5:]
# biowave = biowave[68:243]
# x = range(len(biowave))
#
# print(biosignal)
# print(biowave)
#
# print(x)
#
# # plt.rcParams["figure.figsize"] = (1000,1000)
# plt.rcParams["figure.figsize"] = (70, 70)
# plt.plot(x, biowave)
# plt.show()


count = []
start_point = [0]
end_point = []
# wave_range = 7500-73
wave_range = 6505-1289

for i in range(wave_range):
    if i%175 == 0:
        count.append(i)

print("COUNT :", count)

plt.rcParams["figure.figsize"] = (25, 10)

biosignal = sj1_csv_people1[1:5]
# biowave_full = sj1_csv_people1[73:]
biowave_full = sj1_csv_people1[1289:6505]
biowave_full.index = range(len(biowave_full))
# print("bio",biowave_full.values)
# print(biowave_full[169])
print(biowave_full)

for i in count:
    A = biowave_full[i+50:i+200]
    print(A.values)
    print("i", i)

    if i+200 > len(biowave_full):
        break
    
    # print(A.groupby(A[0]).last())
    print(A)

    min_idx = A.astype(float).idxmin()
    min = A.astype(float).min()

    print("min_idx",min_idx)

    if min_idx != wave_range-1:
        for j in range(1, 5):
            print(j)
            if A[min_idx + j] == min:
                pass
            else:
                min_idx = min_idx + j - 1
                break
    # print("idx :", A.astype(float).idxmin())
    # print("min :", A.min())
    start_point.append(min_idx)
    print(min_idx)
    print("min", min)


    # A = biowave_full[i:i+200]
    # df = pd.DataFrame(A)
    # print(df.columns)
    # print(df.duplicated(0, keep='last'))
    # print(df)
    # A = df[0]
    #
    # print(A)
    #
    # min_idx = A.astype(float).idxmin()
    # print(min_idx)
    # # print("idx :", A.astype(float).idxmin())
    # # print("min :", A.min())
    # start_point.append(min_idx)

end_point = start_point[1:]
end_point.append(wave_range)

print("start", start_point)
print(end_point)

# samp = []

Max_x = 256

# for i, point in enumerate(start_point):
#     biowave = biowave_full[point:end_point[i]+1]
#     x = range(len(biowave))
#
#     # biowave.index
#     #
#     # print(biowave)
#     # samp.append(biowave.astype(float).idxmin())
#     # print("idx :", biowave.astype(float).idxmin())
#     # print("min :", biowave.min())
#
#     # min_idx = biowave.astype(float).idxmin()
#
#     # biowave = biowave_full[i:min_idx]
#     # x = range(min_idx)
#     # # print("x :", x)
#     # plt.plot(x, biowave)
#
#     print("biowave :", biowave)
#     print("x :", x)
#     plt.plot(x, biowave)
#     plt.show()

biowave_filter = biowave_full

min = biowave_filter.astype(float).min()
print("min", min)

for i in range(len(biowave_filter)):
    biowave_filter.iloc[i] = biowave_filter[i] - min

max = biowave_filter.astype(float).max()
print("max", max)

for i in range(len(biowave_filter)):
    biowave_filter.iloc[i] = biowave_filter.iloc[i] / max

for i, point in enumerate(start_point):
    # biowave = pd.Series(biowave_filter[point:end_point[i]+1], index=range(point, end_point[i]+1))
    # biowave = pd.Series(biowave_filter[point:end_point[i]+1], index=range(0, ((end_point[i]+1)-(point))))
    biowave = biowave_filter[point:end_point[i]+1]
    # biowave = biowave.reindex(range(0, ((end_point[i]+1)-(point))))
    # print("i",end_point[i]+1)
    # print("bio", biowave)
    len_biowave = len(biowave)
    Max_x = 256 - len_biowave + 1
    last_index_num = biowave.index[len_biowave-1]
    # print("Max",Max_x)
    # print("len_bio",len_biowave)
    # print("idx", biowave.index[len_biowave-1])
    for j in range(0, Max_x):
        biowave.loc[last_index_num+j] = 0
        # print("j",j)
        # print("len + j",len_biowave + j)
        # print("biowave", biowave.iloc[len_biowave])
        # print("??", biowave.iloc[len_biowave])
        # print(biowave)
    print(biowave.values)
    x = range(len(biowave))
    # print(x)
    plt.rcParams["figure.figsize"] = (18, 7)
    plt.plot(x, biowave)
    plt.axis([0, len(biowave), 0, 1])
    plt.show()
