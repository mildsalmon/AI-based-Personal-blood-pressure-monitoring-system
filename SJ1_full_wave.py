# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import pandas as pd


class SJ1_full_wave:
    def __init__(self, num):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.width', 1000)

        people_num = num

        sj1_csv_people = {
            # people 1
            1: self.select_people(0),
            # people 2
            2: self.select_people(1),
            # people 3
            3: self.select_people(2),
            # people 4
            4: self.select_people(3),
            # people 5
            5: self.select_people(4)
        }

        self.sj1_csv_people_select = sj1_csv_people[people_num]

        print(self.sj1_csv_people_select)

        self.setting(people_num)
        self.filter()
        self.draw()


    def select_people(self, num):
        sj1_csv = pd.read_csv('data/SJ1.csv', engine='python')

        print(sj1_csv)
        #
        # print(sj1_csv.shape)
        #
        # print(sj1_csv.dropna(axis=0, how='any'))

        sj1_csv_exam = sj1_csv.dropna(axis=0, how='any')
        sj1_csv_people = sj1_csv_exam.iloc[num, :]

        return sj1_csv_people

    def setting(self, people_num):
        if people_num == 1:
            wave_start_point = 1289
            wave_end_point = 6505
        elif people_num == 2:
            wave_start_point = 5
            wave_end_point = 0
        elif people_num == 3:
            wave_start_point = 5
            wave_end_point = 0
        elif people_num == 4:
            wave_start_point = 5
            wave_end_point = 0
        elif people_num == 5:
            wave_start_point = 5
            wave_end_point = 0

        self.wave_start_point = wave_start_point
        self.wave_end_point = wave_end_point

    def filter(self):
        biosignal = self.sj1_csv_people_select[1:5]
        if self.wave_end_point == 0:
            biowave = self.sj1_csv_people_select[self.wave_start_point:]
        elif self.wave_end_point != 0:
            biowave = self.sj1_csv_people_select[self.wave_start_point:self.wave_end_point]
        x = range(len(biowave))

        print("biosignal :",biosignal)
        print("biowave :", biowave)
        print("x :", x)

        min = biowave.astype(float).min()

        print("min :", min)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] - min

        max = biowave.astype(float).max()

        print("max :", max)

        for i in range(len(biowave)):
            biowave.iloc[i] = biowave.iloc[i] / max

            print(i, " ", biowave.iloc[i])

        x = range(len(biowave))

        self.x = x
        self.biowave = biowave

    def draw(self):
        plt.rcParams["figure.figsize"] = (25, 10)
        plt.plot(self.x, self.biowave)
        plt.axis([0, len(self.biowave), 0, 1])
        plt.show()

if __name__ == "__main__":

    SJ1 = SJ1_full_wave(2)
    SJ2 = SJ1_full_wave(3)
    SJ1 = SJ1_full_wave(4)
    SJ1 = SJ1_full_wave(5)
