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

print(sj1_csv_people1)


biosignal = sj1_csv_people1[1:5]
# biowave = sj1_csv_people1[5:6675]

# people1의 wave 30개 선정
biowave = sj1_csv_people1[1289:6505]
x = range(len(biowave))

print(biosignal)
print(biowave)

print(x)

# plt.rcParams["figure.figsize"] = (1000,1000)
plt.rcParams["figure.figsize"] = (25, 10)
plt.plot(x, biowave)
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

plt.plot(x, biowave)
plt.axis([0, len(biowave), 0, 1])
plt.show()

min = biowave.astype(float).min()
max = biowave.astype(float).max()
print("min", min)
print("max", max)