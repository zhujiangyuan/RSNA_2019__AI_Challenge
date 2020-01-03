"""
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview/hemorrhage-types
Five subtypes,"any" means images can fall into any category.
"""
Hemorrhage_Subtypes = [
    'epidural',
    'intraparenchymal',
    'intraventricular',
    'subarachnoid',
    'subdural',
    'any']

Subtypes_Number = len(Hemorrhage_Subtypes)

"""
Refer to stage_2_train.csv, there are 4516842 labels.
Each sample has 6 labels.
"""
All_Train_labels = 4516842
All_Train_Samples = 752807
Minimum_Samples = 3
Train_CSV_File_Name = "stage_2_train.csv"

DCM_Image_Suffix = ".dcm"
