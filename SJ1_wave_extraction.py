# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import pandas as pd

class Wave_Extraction:
    def __init__(self, num):
        # 판다스 크기 키우기
        # pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)

        self.count = []
        self.start_point = [0]
        self.end_point = []

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
        for i in range(self.wave_range):
            if i % 175 == 0:
                self.count.append(i)

        print("COUNT : ", self.count)

    def biosignal(self):
        recode_time = self.people_wave[0]
        bio_signal = self.people_wave[1:5]
        self.bio_wave_full = self.people_wave[self.total_start:self.total_end]
        self.bio_wave_full.index = range(len(self.bio_wave_full))

        print("Bio_Wave_Full : ", self.bio_wave_full)

    def set_one_wave(self):
        for i in self.count:
            A = self.bio_wave_full[i + 50 : i + 195] # ?
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
            self.start_point.append(min_idx)
            print(min_idx)
            print("min : ", min)

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

    def draw_wave(self):
        for i, point in enumerate(self.start_point):
            biowave = self.bio_wave_full[point:self.end_point[i] + 1]
            len_biowave = len(biowave)
            Max = self.Max_x - len_biowave + 1
            last_index_num = biowave.index[len_biowave - 1]

            for j in range(0, Max):
                biowave.loc[last_index_num + j] = 0

            print(biowave.values)
            x = range(len(biowave))

            plt.rcParams["figure.figsize"] = (18, 7)
            plt.plot(x, biowave)
            plt.axis([0, len(biowave), 0, 1])
            # plt.show()
            plt.savefig('image/wave_{0}.png'.format(i))

            plt.cla()


if __name__ == "__main__":
    Wave_Extraction(2)