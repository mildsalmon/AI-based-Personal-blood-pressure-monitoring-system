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
sj1_csv_people5 = sj1_csv_exam.iloc[5, :]

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
wave_range = 7500-73

for i in range(wave_range):
    if i%175 == 0:
        count.append(i)

print("COUNT :", count)

plt.rcParams["figure.figsize"] = (25, 10)

biosignal = sj1_csv_people1[1:5]
biowave_full = sj1_csv_people1[73:]
biowave_full.index = range(len(biowave_full))

# print(biowave_full[169])
# print(biowave_full)

for i in count:
    A = biowave_full[i+50:i+200]
    print(A.values)

    
    # print(A.groupby(A[0]).last())
    print(A)

    min_idx = A.astype(float).idxmin()
    min = A.astype(float).min()
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
    print(min)


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

print(start_point)
print(end_point)

# samp = []

for i, point in enumerate(start_point):
    biowave = biowave_full[point:end_point[i]+1]
    x = range(len(biowave))

    # biowave.index
    #
    # print(biowave)
    # samp.append(biowave.astype(float).idxmin())
    # print("idx :", biowave.astype(float).idxmin())
    # print("min :", biowave.min())

    # min_idx = biowave.astype(float).idxmin()

    # biowave = biowave_full[i:min_idx]
    # x = range(min_idx)
    # # print("x :", x)
    # plt.plot(x, biowave)

    print("biowave :", biowave)
    print("x :", x)
    plt.plot(x, biowave)
    plt.show()

# print(samp)