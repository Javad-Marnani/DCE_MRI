##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import pandas as pd
import numpy as np
def path_provider():
    """
    Provides paths for various directories and files related to the BC_MRI dataset.

    Returns:
        A tuple containing the following paths:
        - current_path: The current working directory.
        - bc_mri_path: The path to the BC_MRI directory.
        - dataset_path: The path to the dataset directory within BC_MRI.
        - xlsx_csv_files_path: The path to the xlsx_csv_files directory within BC_MRI.
        - samples_path: The path to the Duke-Breast-Cancer-MRI directory within the dataset.
        - clinical_file_path: The path to the Clinical_and_Other_Features.csv file within xlsx_csv_files.
        - mapping_path: The path to the Breast-Cancer-MRI-filepath_filename-mapping.csv file within xlsx_csv_files.
        - boxes_path: The path to the Annotation_Boxes.csv file within xlsx_csv_files.
        - radiomics_clinical_path: The path to the radiomics_clinical_features_data.csv file within BC_MRI/extracted_features.
        - features_by_saha: The path to the Imaging_Features.csv file within xlsx_csv_files.
    """
    # current_path = os.path.dirname(os.path.abspath(__file__))
    current_path = os.getcwd()
    bc_mri_path = current_path + r'\BC_MRI'
    dataset_path = bc_mri_path + r'\dataset'
    xlsx_csv_files_path = bc_mri_path + r'\xlsx_csv_files'
    samples_path = dataset_path + r'\Duke-Breast-Cancer-MRI'
    directories_to_check = ["dataset", "extracted_features", "resized_images", "xlsx_csv_files"]
    for folder in directories_to_check:
        folder_path = os.path.join(bc_mri_path, folder)
        if not os.path.exists(folder_path):
           os.makedirs(folder_path)
    clinical_file_path = xlsx_csv_files_path + r'\Clinical_and_Other_Features.csv'
    mapping_path = xlsx_csv_files_path + r'\Breast-Cancer-MRI-filepath_filename-mapping.csv'
    boxes_path = xlsx_csv_files_path + r'\Annotation_Boxes.csv'
    radiomics_clinical_path=bc_mri_path+r'\extracted_features\radiomics_clinical_features_data.csv'
    features_by_saha=xlsx_csv_files_path + r'\Imaging_Features.csv'
    return current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha


def covert_xlsx_to_csv():
    """
    Converts XLSX files to CSV format and saves them in the designated 'xlsx_csv_files' directory.

    Uses the path_provider() function to obtain the necessary file paths.

    The XLSX files to be converted are:
    - Breast-Cancer-MRI-filepath_filename-mapping.xlsx
    - Annotation_Boxes.xlsx
    - Clinical_and_Other_Features.xlsx
    - Imaging_Features.xlsx

    The function iterates over the XLSX and CSV file paths, checks if the corresponding CSV file already exists,
    and converts the XLSX file to CSV format using pandas. The resulting CSV data is then written to the CSV file.
    If a CSV file already exists, that specific conversion is skipped.
    """
    current_path, bc_mri_path, dataset_path, xlsx_csv_files_path, samples_path, clinical_file_path, mapping_path, boxes_path, radiomics_clinical_path, features_by_saha = path_provider()
    # List of XLSX file paths
    xlsx_file_paths = [xlsx_csv_files_path+ r'\Breast-Cancer-MRI-filepath_filename-mapping.xlsx',
                   xlsx_csv_files_path+ r'\Annotation_Boxes.xlsx',
                   xlsx_csv_files_path+ r'\Clinical_and_Other_Features.xlsx',
                   xlsx_csv_files_path+ r'\Imaging_Features.xlsx'
                      ]
    # Convert XLSX file paths to CSV file paths
    csv_file_paths = [path[:-4] + "csv" for path in xlsx_file_paths]
    # Iterate over the XLSX and CSV file paths
    for xlsx_file, csv_file in zip(xlsx_file_paths, csv_file_paths):
        # Check if the CSV file already exists
        if os.path.exists(csv_file):
            # If the CSV file exists, skip this iteration
            continue
        print("Converting XLSX files to CSV format and saving them in the designated 'xlsx_csv_files' directory.")
        # Read the XLSX file into a pandas DataFrame
        data_frame = pd.read_excel(xlsx_file)

        # Convert DataFrame to CSV format
        csv_data = data_frame.to_csv(index=False)

        # Write CSV data and save it in the target directory
        with open(csv_file, "w") as file:
            file.write(csv_data)
