# -*- coding: utf-8 -*-

import pandas as pd
import os


class Collection:
    def __init__(self, isNew):
        """
        :param isNew:
            True = 장비를 통해 새로 측정한 SpO2 Wave만 전처리 한다.
            False = 전체 데이터를 새로 전처리한다. (전처리 방식이 변한 경우)
        """
        self.new = isNew

        self.write_New_OR_Old()

    def write_new_or_old(self):
        """
        파일명들이 저장된 경로 설정 및 self.new = isNew의 값에 따라 다른 파일명을 저장하도록 메소드를 따로 호출함.
        :return:
        """
        # read_path = 장비로 측정한 데이터 경로
        # write_path = 전처리 후 저장하는 데이터 경로
        self.read_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\수집")
        self.write_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\수집\save")

        print(self.read_path)
        print(self.write_path)

        # read_path_list => save 딕셔너리를 제외한 파일명 리스트
        self.read_path_list = os.listdir(self.read_path)[1:]
        self.write_path_list = os.listdir(self.write_path)

        print(self.read_path_list)
        print(self.write_path_list)

        # 객체 생성 초기에 장비를 통해 새로 측정한 데이터만 전처리할 것인지,
        # 전체 데이터를 새로 전처리할 것인지 구분하여 메소드 호출
        if self.new == True:
            self.Make_New_CSV()
        elif self.new == False:
            self.Edit_Old_CSV()

    def make_new_csv(self):
        """
        새로운 SpO2 Wave 값만 전처리 할 수 있도록 새로운 파일만 체크해서 리스트에 추가함.
        :return:
        """
        new_list = []

        for csv_name in self.read_path_list:
            if csv_name not in self.write_path:
                new_list.append(csv_name)

        self.run(new_list)

    def edit_old_csv(self):
        """
        모든 파일을 전처리하기 위해 기존에 있는 파일명을 리스트에 추가함.
        :return:
        """
        new_list = self.read_path_list

        self.run(new_list)

    def run(self, csv_name_list):
        """
        전처리하는 메소드

        :param csv_name_list:
            전처리에 사용할 csv 파일명 리스트
        :return: 
        """


if __name__ == "__main__":
    Collection(isNew=True)