##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import cv2 as cv
import collections
import numpy as np
import pandas as pd
#import tensorflow as tf
from Feature_Extraction import Feature_Extraction
from Data_Preprocessing import Data_Preprocessing
from Binary_Classifications_OvR_OvO import Binary_Classifications_OvR_OvO
from Radiomics_Features_Saha import Radiomics_Features_Saha
##########################################################################################
####################################   SETTINGS    #######################################
##########################################################################################
_GPU = False

##########################################################################################
######################################   TEMP    #########################################
##########################################################################################
pd.set_option('display.max_columns', None)
##########################################################################################
#################################   GPU CONFIGURATION AND SELECTION    #####################################
##########################################################################################
'''Choose GPU 0 or 1 if they are available for processing.'''
if _GPU:
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
  tf.config.set_visible_devices(physical_devices[1], 'GPU')
  visible_devices = tf.config.get_visible_devices('GPU')
  print(visible_devices)
##########################################################################################
################################   main   ##################################
##########################################################################################

def main():
  Feature_Extraction()
  Data_Preprocessing()
  Binary_Classifications_OvR_OvO()
  Radiomics_Features_Saha()

if __name__ == '__main__':
    main()

