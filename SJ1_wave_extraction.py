# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import pandas as pd

class Wave_Extraction:
    def __init__(self, num):
        # 판다스 크기 키우기
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)

        self.count = []
        self.start_point = [0]
        self.end_point = []

        self.bio_wave_list = pd.DataFrame()
        self.num = num

        self.Max_x = 256

        self.file_read()
        self.people_wave = self.preprocessing(num)

        if self.people_wave.empty:
            pass
        else:
            self.set_count()
            self.biosignal()
            self.set_one_wave()
            self.one_wave_normalization()
            self.draw_wave()


    def file_read(self):
        self.sj1_csv = pd.read_csv('data/SJ1.csv', engine='python')

        print(self.sj1_csv)
        print(self.sj1_csv.shape)
        print(self.sj1_csv.dropna(axis=0, how='any'))

    def preprocessing(self, num):
        sj1_csv_exam = self.sj1_csv.dropna(axis=0, how='any')

        if num == 1:
            self.sj1_csv_people = sj1_csv_exam.iloc[0, :]
            total_start = 753
            total_end = 6825
        elif num == 2:
            self.sj1_csv_people = sj1_csv_exam.iloc[1, :]
            total_start = 179
            total_end = 6643
        elif num == 3:
            self.sj1_csv_people = sj1_csv_exam.iloc[2, :]
            total_start = 1557
            total_end = 7331
        elif num == 4:
            self.sj1_csv_people = sj1_csv_exam.iloc[3, :]
            total_start = 821
            total_end = 7396
        elif num == 5:
            self.sj1_csv_people = sj1_csv_exam.iloc[4, :]
            total_start = 772
            total_end = 7351
        else:
            # return 0
            pass
        print(self.sj1_csv_people)

        wave_range = total_end - total_start

        self.total_end = total_end
        self.total_start = total_start
        self.wave_range = wave_range

        return self.sj1_csv_people

    def set_count(self):
        # for i in range(self.wave_range):
        #     if i % 175 == 0:
        #         self.count.append(i)
        #
        # print("COUNT : ", self.count)

        self.count = [0]

    def biosignal(self):
        recode_time = self.people_wave[0]
        bio_signal = self.people_wave[1:5]
        self.bio_wave_full = self.people_wave[self.total_start:self.total_end]
        self.bio_wave_full.index = range(len(self.bio_wave_full))

        self.BP = bio_signal[2:4]
        # 판다스 인덱스는 숫자만 인식
        self.BP.index = [-2, -1]
        print("BP",type(self.BP))
        print("BP",self.BP)

        print("Bio_Wave_Full : ", self.bio_wave_full)

    def set_one_wave(self):
        for i in self.count:
            if i < self.wave_range:
                A = self.bio_wave_full[i + 50 : i + 250] # ?
                print("A Value : ", A.values)
                print("i : ", i)

                if i + 200 > len(self.bio_wave_full):
                    break

                print(A)

                min_idx = A.astype(float).idxmin()
                min = A.astype(float).min()

                print("min_idx", min_idx)

                if min_idx != self.wave_range - 1:
                    for j in range(1, 5):
                        print(j)

                        if self.bio_wave_full[min_idx + j] == min:
                            pass
                        else:
                            min_idx = min_idx + j - 1
                            break
                self.count.append(min_idx)
                print("CounT : ", self.count)
                self.start_point.append(min_idx)
                print(min_idx)
                print("min : ", min)
            else:
                break

        self.end_point = self.start_point[1:]
        self.end_point.append(self.wave_range)

        print("start : ", self.start_point)
        print("end : ", self.end_point)

    def one_wave_normalization(self):
        min = self.bio_wave_full.astype(float).min()
        print("min : ", min)

        for i in range(len(self.bio_wave_full)):
            self.bio_wave_full.iloc[i] = self.bio_wave_full[i] - min

        max = self.bio_wave_full.astype(float).max()
        print("max : ", max)

        for i in range(len(self.bio_wave_full)):
            self.bio_wave_full.iloc[i] = self.bio_wave_full.iloc[i] / max

        print("bio_wave_full : ", self.bio_wave_full[179])

    def draw_wave(self):
        bio_index = []
        for k in range(256):
            bio_index.append(k)

        for i, point in enumerate(self.start_point):
            biowave = self.bio_wave_full[point:self.end_point[i] + 1]
            print(point)
            print(self.bio_wave_full[point])
            print(self.bio_wave_full[175:185])
            print("biowave : ", type(biowave))
            print("biowave : ", biowave)
            len_biowave = len(biowave)
            Max = self.Max_x - len_biowave + 1
            last_index_num = biowave.index[len_biowave - 1]

            print("last_index_num : ", last_index_num)

            for j in range(1, Max):
                biowave.loc[last_index_num + j] = 0

            print("bi",biowave.index)

            biowave.index = bio_index

            print(biowave.values)
            print(biowave.index)

            self.one_wave = biowave
            x = range(len(biowave))

            self.bio_wave_con = pd.concat([self.BP, self.one_wave])
            print("Te",type(self.bio_wave_con))

            # self.bio_wave_list = self.bio_wave_list.append(self.BP, ignore_index=True)
            self.bio_wave_list = self.bio_wave_list.append(self.bio_wave_con, ignore_index=True)
            # print("One Bio wave\n",i,"\n", self.bio_wave_list)

            plt.rcParams["figure.figsize"] = (18, 7)
            plt.plot(x, biowave)
            plt.axis([0, len(biowave), 0, 1])
            # plt.show()
            print("I:",i)
            plt.savefig('image/wave_{0}.png'.format(i))


            # plt.cla()

        # self.save_csv()

        print("One Bio wave\n", self.bio_wave_list)

    def save_csv(self):
        print("one",type(self.one_wave))
        # self.bio_wave_list.to_csv("data/people.csv", mode='a')
        self.bio_wave_list.to_csv("data/people.csv", mode='a', header=False)

if __name__ == "__main__":
    # Wave_Extraction(1)
    # Wave_Extraction(2)
    # Wave_Extraction(3)
    # Wave_Extraction(4)
    Wave_Extraction(5)
