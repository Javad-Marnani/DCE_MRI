##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
from Images_Functions import random_sample_for_each_cancer_type, filter_mapping_df,save_dcm_slice,process_mapping_df,image_filenames_plot_one_at_random,patients_number
from Images_Functions import subtype_frequency,count_patient_slices,process_paths,sort_slices,sort_paths,plot_cropped_images,crop_and_save_images,extract_pixels
from Images_Functions import plot_images,Include,threshold_segmentation,feature_extraction
from Path_Functions import path_provider,covert_xlsx_to_csv
####################################   SETTINGS    #######################################
##########################################################################################
_GPU = False

##########################################################################################
######################################   TEMP    #########################################
##########################################################################################
pd.set_option('display.max_columns', None)

##########################################################################################
#################################   GPU CONFIGURATION AND SELECTION     #####################################
##########################################################################################
'''Choose GPU 0 or 1 if they are available for processing.'''
if _GPU:
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
  tf.config.set_visible_devices(physical_devices[1], 'GPU')
  visible_devices = tf.config.get_visible_devices('GPU')
  print(visible_devices)
##########################################################################################
#################################   Feature_Extraction     #####################################
##########################################################################################
def Feature_Extraction():
  current_path,bc_mri_path,dataset_path,xlsx_csv_files_path,samples_path,clinical_file_path,mapping_path,boxes_path,radiomics_clinical_path,features_by_saha=path_provider()
  print("Current path is: ", current_path)
  print("Please ensure that you have placed four XLSX files into the 'xlsx_csv_files' directory before running the process.")
  covert_xlsx_to_csv()
  types = ['pre', 'post_1', 'post_2', 'post_3']
  # Call the function to generate random samples for each cancer type
  global list0, list1, list2, list3
  list0, list1, list2, list3 = random_sample_for_each_cancer_type(clinical_file_path)
  # Combine the first elements from each list into a single list
  global list_
  list_ = list0 + list1 + list2 + list3
  # Print the length of the combined list
  print("The total sample size is:", len(list_))
  # Print the selected patients
  print("Please download the following patient folders and put the folder Duke-Breast-Cancer-MRI in dataset directory:")
  for folder in sorted(list_):
      print(folder)
  print("Separating the cancer-containing slices (pos) from the non-cancer-containing slices (neg) and saving them in different directories.")
  for seq_type in types:
      print("Processing", seq_type, "contrast sequence ")
      mapping_df=filter_mapping_df(mapping_path, list_, seq_type)
      process_mapping_df(dataset_path, mapping_path, boxes_path, mapping_df, seq_type)
  #Number of all cancer-containing slices derived from the selected patients
  n = len(image_filenames_plot_one_at_random(seq_type='pre', label='pos', show=False))
  print("Number of all cancer-containing slices derived from the selected patients is:",n)
  #Frequencies of cancer-containing slices for each subtype
  n0=subtype_frequency(subtype='Luminal A',seq_type='pre', label='pos')
  n1=subtype_frequency(subtype='Luminal B',seq_type='pre', label='pos')
  n2=subtype_frequency(subtype='HER2+',seq_type='pre', label='pos')
  n3=subtype_frequency(subtype='TN',seq_type='pre', label='pos')
  print("Frequencies of cancer-containing slices for each subtype are as follow:")
  print('Luminal A:',n0,'\nLuminal B:',n1,'\nHER2+:',n2,'\nTN:',n3)
  print("Dictionary containing the count of cancer-containing slices for each patient:")
  print(count_patient_slices(seq_type='post_3', label='pos'))
  plot_cropped_images(boxes_path,seq_type='pre', label='pos', show=False)
  print("Cropping and saving the images if they don't already exist in the 'resized_images' directory.")
  for seq_type in tqdm(['pre', 'post_1', 'post_2', 'post_3']):
      for crop in  tqdm(['original',32,64]):
          crop_and_save_images(boxes_path,seq_type, crop,label='pos', show=False)
  #Extract pixels from images and store them in a matrix
  pixel = extract_pixels(seq_type='post_2',crop='original')
  # Print the shape of the pixel matrix
  print("shape of the pixel matrix is:", np.shape(pixel))
  # Plot the images from the 'pixel' matrix
  plot_images(seq_type='pre',crop='original', num_images=False)
  # Calculate the number of included slices in the 'include' array (s=3)
  include=Include(seq_type='pre', label='pos', s=3)
  u, c = np.unique(include, return_counts=True)
  print("Number of included (1) and excluded (0) slices in the include array (s=3):\n",pd.Series(c, index=u))
  # Perform threshold segmentation on the 'pixel' array with a threshold of 50.
  mask = threshold_segmentation(seq_type='post_1',crop=64, threshold=50, show=False)
  # Print the shape of the 'mask' array
  print("The shape of generated masks:\n",mask.shape)
  print("==============================================")
  print("||   FEATURE EXTRACTION IN PROGRESS...       ||")
  print("==============================================")
  # Iterate over seq_type and crop values
  for seq_type in tqdm(['pre', 'post_1', 'post_2', 'post_3']):
      for crop in tqdm(['original', 32, 64]):
          print(f"Generating CSV file associated with sequence {seq_type} and crop {crop}")
          # Check if the CSV file already exists
          filename = f"{bc_mri_path}/extracted_features/{seq_type.capitalize()}_{str(crop).capitalize()}.csv"
          if os.path.exists(filename):
              # If the CSV file exists, skip this iteration
              continue
          # Obtain the data using the radiomics features
          data = feature_extraction(seq_type, crop, threshold=25, s=3, label='pos')
          D = pd.DataFrame(data=data)
          # Select rows where all columns are zero (related to the slices that were not selected)
          all_zeros = (D == 0).all(axis=1)
          # Drop the rows where all columns are zero
          D = D[~all_zeros]
          print("Total number of missing values:", D.isnull().sum().sum())
          # Write the data and save it as a CSV file
          D.to_csv(filename, header=True)
  #Load the CSV file 'Pre_Original.csv' into a pandas DataFrame
  D = pd.read_csv(bc_mri_path+'/extracted_features/Pre_Original.csv')
  #Remove the first column from the DataFrame
  D = D.iloc[:, 1:]
  #Display the updated DataFrame
  #print("The data frame for pre contrast sequences and original crop is:")
  #display(D)
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - FEATURE EXTRACTION DONE  - - - - - - - - - - - - - - - -')
  print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
if __name__ == '__main__':
  Feature_Extraction()