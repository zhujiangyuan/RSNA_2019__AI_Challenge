import joblib
import PIL
import pydicom
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import argparse


class DataPreprocessing:

    __Image_Suffix = ".png"

    @staticmethod
    def DataPreprocessingParallel(img_paths, output_folder_path, n_jobs=-1):
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(DataPreprocessing.__DataPreprocessingImp)(i, output_folder_path) for i in tqdm(img_paths)
        )

    @staticmethod
    def DataPreprocessing(img_folder_path, sub_folder):
        for i in tqdm.tqdm(img_folder_path):
            DataPreprocessing.__ProcessingSingleImage(i, sub_folder)

    @classmethod
    def SetMetaData(cls, meta_data):
        cls.__meta_data = meta_data

    @staticmethod
    def __DataPreprocessingImp(img_path, sub_folder):
        try:
            img_id, img_buffer = DataPreprocessing.__ProcessingSingleImage(img_path)
            DataPreprocessing.__SaveImg(img_buffer, sub_folder, img_id)
        except KeyboardInterrupt:
            # Rais interrupt exception so we can stop the cell execution
            # without shutting down the kernel.
            raise

    @staticmethod
    def __ProcessingSingleImage(img_path, img_type=np.int8):
        img_dicom = pydicom.read_file(img_path)
        img_id = DataPreprocessing.__GetSOPInstanceUID(img_dicom)
        meta_data = DataPreprocessing.__GetMetaData(img_dicom)
        img = DataPreprocessing.__AdjustWWWL(img_dicom.pixel_array, **meta_data)
        img = DataPreprocessing.__ImageNormalize(img) * 255
        img = PIL.Image.fromarray(img.astype(img_type), mode="L")
        return img_id, img

    @staticmethod
    def __ImageNormalize(img):
        min_value, max_value = img.min(), img.max()
        return (img - min_value) / (max_value - min_value)

    @staticmethod
    def __AdjustWWWL(img, window_center, window_width, intercept, slope):
        img = img * slope + intercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    @staticmethod
    def __GetSOPInstanceUID(img_dicom):
        return str(img_dicom.SOPInstanceUID)

    @staticmethod
    def __GetTagValue(x):
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        return int(x)

    @staticmethod
    def __GetMetaData(img_dicom):
        """
        Hard to modify or extend.
        """
        metadata = {
            "window_center": img_dicom.WindowCenter,
            "window_width": img_dicom.WindowWidth,
            "intercept": img_dicom.RescaleIntercept,
            "slope": img_dicom.RescaleSlope,
        }
        return {k: DataPreprocessing.__GetTagValue(v) for k, v in metadata.items()}

    @staticmethod
    def __ImageResize(img, width_height, image_type=np.int8):
        img = PIL.Image.fromarray(img.astype(image_type), mode="L")
        return img.resize(width_height, resample=PIL.Image.BICUBIC)

    @staticmethod
    def __SaveImg(img_pil, sub_folder, name):
        save_image_full_path = os.path.join(sub_folder, ame + DataPreprocessing.__Image_Suffix)
        img_pil.save(save_image_full_path)


if __name__ == '__main__':
    """"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dcm_path", "--dcm_path", type=str)
    parser.add_argument("-png_path", "--png_path", type=str)
    args = parser.parse_args()
    dcm_path = args.dcm_path
    png_path = args.png_path

    if not os.path.exists(png_path):
        os.makedirs(png_path)

    prepare_images_njobs(glob(dcm_path+'/*'), png_path+'/')
    """
    print(DataPreprocessing.static_name)
    print("OK")
