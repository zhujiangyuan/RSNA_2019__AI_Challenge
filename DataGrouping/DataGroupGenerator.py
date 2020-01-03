from Utility import Constant
import os
import pandas as pd
import numpy as np
import threading
import time
import shutil
import random

class DataGroupGenerator:
    """ This class is used to create a data group. """

    def __init__(self,
                 src_image_folder_path=None,
                 target_image_folder_full_path=None,
                 train_csv_file_name=Constant.Train_CSV_File_Name,
                 file_path=None,
                 sample_number=None,
                 random_seed=None
                 ):

        """ Get train CSV file path. """
        if not isinstance(train_csv_file_name, str):
            raise Exception("Invalid train_csv_file_name!", train_csv_file_name)
        if file_path is None:
            """ Use relative path. """
            path = os.getcwd()
            self.__train_csv_file_full_path = os.path.join(path, train_csv_file_name)
        else:
            self.__train_csv_file_full_path = os.path.join(file_path, train_csv_file_name)
        if not os.path.exists(self.__train_csv_file_full_path):
            raise Exception("Invalid train csv file path !", self.__train_csv_file_full_path)

        if src_image_folder_path is None:
            self.__src_image_folder_path = os.getcwd()
        else:
            if not os.path.exists(src_image_folder_path):
                raise Exception("Invalid src image folder path!", src_image_folder_path)
            self.__src_image_folder_path = src_image_folder_path

        if target_image_folder_full_path is None:
            self.__target_image_folder_path = self.__src_image_folder_path
        else:
            if not os.path.exists(target_image_folder_full_path):
                os.makedirs(target_image_folder_full_path)
            self.__target_image_folder_full_path = target_image_folder_full_path

        if not os.path.isdir(self.__target_image_folder_full_path):
            os.makedirs(self.__target_image_folder_full_path)

        """ Get sample number. """
        if sample_number is not None:
            if not isinstance(sample_number, int):
                raise Exception("Invalid sample number!", sample_number)
            if sample_number > Constant.All_Train_Samples or sample_number < Constant.Minimum_Samples:
                raise Exception("Invalid sample number! \n "
                                "Max sample size is ", Constant.All_Train_Samples)
        self.__sample_number = sample_number

        if random_seed is not None:
            max_seed = Constant.All_Train_Samples - self.__sample_number
            self.__start_sample_index = random.randint(0, max_seed)

        self.__data_info = []

        self.__ReadCSVFile()
        self.__SaveCSV()
        self.__GenerateDataInfo()

    def __ReadCSVFile(self):
        rows = None
        if self.__sample_number is not None:
            rows = self.__sample_number * Constant.Subtypes_Number
        skip_rows = self.__start_sample_index * Constant.Subtypes_Number
        self.__csv_data = pd.read_csv(self.__train_csv_file_full_path, index_col=False, skiprows=skip_rows, nrows=rows)

    __Sub_Thread_threshold = 1000
    __Default_Thread_Number = 5

    def __GenerateDataInfo(self, thread_number=__Default_Thread_Number):
        row_number = len(self.__csv_data)
        block_list = []
        if row_number > DataGroupGenerator.__Sub_Thread_threshold:
            block_size = row_number // thread_number
            for i in range(thread_number):
                start = i * block_size
                if i == thread_number - 1:
                    end = row_number
                else:
                    end = (i + 1) * block_size - 1
                group = (start, end)
                block_list.append({"func": DataGroupGenerator.__threadFunc, "args": (self, group,)})
        else:
            group = (0, row_number)
            block_list.append(group)
            return self.__threadFunc(block_list[0])

        self.__data_lock = threading.Lock()
        data_collect_threads = DataGroupGenerator.DataCollectThread(block_list)
        data_collect_threads.start()
        print(len(self.__data_info))

    def __threadFunc(self, index_tuple):
        start = index_tuple[0]
        print(start)
        end = index_tuple[1]
        """ iterrows is slow """
        block_data = np.array(self.__csv_data.loc[start:end])
        block_data_info = []
        row_number = len(block_data)
        image_set = set()
        for i in range(row_number):
            current_id = block_data[i][0]
            current_label = block_data[i][1]
            image_path = self.GetImageFullPath(current_id)
            image_set.add(image_path)
        for image in image_set:
            shutil.copy(image, self.__target_image_folder_full_path)

    def GetImageFullPath(self, id, split_mark='_'):
        # id:ID_c3b373f81_subdural
        right_mark_position = id.rfind(split_mark)
        if right_mark_position == -1:
            raise Exception("Invalid ID!", id)
        file_name = id[0:right_mark_position]
        return self.__src_image_folder_path + "\\" + file_name + Constant.DCM_Image_Suffix

    @staticmethod
    def __CopyFiles(source_files, dist):
        for file in source_files:
            shutil.copy(file, dist)

    def __SaveCSV(self):
        csv_full_path = os.path.join(self.__target_image_folder_full_path, "label.csv")
        self.__csv_data.to_csv(csv_full_path, header=0)

    class DataCollectThread(object):
        """ THis thread is used for collecting data info. """

        def __init__(self, func_list=None):
            self.ret_flag = 0
            self.func_list = func_list
            self.threads = []

        def start(self):
            self.threads = []
            self.ret_flag = 0
            for func_dict in self.func_list:
                t = threading.Thread(target=func_dict["func"], args=func_dict["args"])
                self.threads.append(t)
            for thread_obj in self.threads:
                thread_obj.start()
            for thread_obj in self.threads:
                thread_obj.join()


if __name__ == '__main__':
    data_group = DataGroupGenerator(src_image_folder_path="G:\\rsna-intracranial-hemorrhage-detection\\stage_2_train",
                                    target_image_folder_full_path="G:\\Test",
                                    sample_number=1000,
                                    random_seed='random')
    print("OK!")
