# DCE-MRI
# Machine Learning-based Prediction of Molecular Subtypes of Breast Cancer using DCE-MRI

This readme file provides instructions for accessing and utilizing the data to predict four molecular subtypes of breast cancer using DCE-MRI and machine learning algorithms.

## Dataset

The dataset can be accessed through this link: [Dataset Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). Download the following data components from the Data Access tab:

- Images (DICOM, 368.4 GB)
- File Path mapping tables (XLSX, 49.6 MB)
- Clinical and Other Features (XLSX, 582 kB)
- Annotation Boxes (XLSX, 49 kB)
- Imaging features (XLSX, 6.44 MB)

## Instructions

1. To filter and download the images, please use the NBIA Data Retriever tool. You can find more information about downloading TCIA images using NBIA Data Retriever at this link: [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Ideally, you should download all the images (922 patient folders) to be able to run the codes for different samples. However, if you have storage limitations, you can download a subset of 200 patient folders, which are the output of the function random_sample_for_each_cancer_type in 1. Feature_Selection.py. Select the "Classic Directory Name" option in NBIA Data Retriever.

2. Instead of converting all the downloaded image data to a single RAR format, it is recommended to convert each patient folder into a separate RAR file based on the labels. This approach will make it easier for uploading the data due to the high volume of images. Create four RAR files labeled as "Label0.rar," "Label1.rar," "Label2.rar," and "Label3.rar" to correspond with the different labels. Finally, upload the respective RAR files to the desired directory.

3. Organize your data by setting up the following folders:

     - Create a "DATA" folder to store all the required raw datasets.
     - Within the main directory, create a "Resized_Images" folder to store the resized images in PNG format.
     - Within the main directory, create an "Extracted Features" folder to store the extracted radiomics features.
4. After extracting the RAR file(s), navigate to the "Balanced_Data" folder within the DATA folder. Create a folder called "Duke-Breast-Cancer-MRI" and place all the extracted patient folders alongside the "metadata.csv" file.

5. Convert all the Excel files (components 2 to 5) to CSV format. To get an understanding of the general data format, you can refer to the provided Dataset folder in this repository, which includes demo data and showcases the expected structure of the data.
 
6. Create and activate a Python or Conda virtual environment and install all the required packages using pip install -r requirements.txt.

7. Start with the "1. Feature_Extraction.py" file to extract 12 combinations of radiomics features. After obtaining the 12 CSV data files, proceed to run the other .py files in the order their names represent. The objectives of each file are as follows: adding clinical features, performing initial feature selection, and classifying molecular subtypes of BC using two approaches:
     - One vs. the rest classifications (4 OvR classifications)
     - One vs. one classifications (6 OvO classifications)
     
   Finally, we want to apply the same methodology for the extracted radiomics features by Saha et al. and make comparisons.
   [Saha et al. - British Journal of Cancer](https://www.nature.com/articles/s41416-018-0185-8)

9. Make any necessary modifications to paths and directories within the code files.
