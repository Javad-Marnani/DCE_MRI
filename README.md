# Machine Learning-based Prediction of Molecular Subtypes of Breast Cancer using DCE-MRI

This README file provides instructions for accessing and utilizing the data to predict four molecular subtypes of breast cancer using DCE-MRI and machine learning algorithms.

## Dataset

The dataset can be accessed through this link: [Dataset Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). Download the following data components from the Data Access tab:

a) Images (DICOM, 368.4 GB)

b) File Path mapping tables (XLSX, 49.6 MB)

c) Clinical and Other Features (XLSX, 582 kB)

d) Annotation Boxes (XLSX, 49 kB)

e) Imaging features (XLSX, 6.44 MB)


## Instructions

1. Use the NBIA Data Retriever tool to filter and download the images. More information about downloading TCIA images using NBIA Data Retriever can be found at this link: [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Ideally, download all the images (922 patient folders) to run the codes for different samples. However, for storage limitations, a subset of 200 patient folders or fewer can be downloaded, which are the output of the first function (`random_sample_for_each_cancer_type`) in `Feature_Extraction.py`.

2. Select the "Classic Directory Name" option in NBIA Data Retriever while downloading the images. After downloading all the patient folders, convert the `Duke-Breast-Cancer-MRI` into a RAR file to ease the upload. The name of the RAR file should be `Duke-Breast-Cancer-MRI.rar` and should not be changed.

3. All required directories will be created automatically if they do not exist. The main directory is `BC_MRI` created in the current path. Three other directories including `dataset`, `resized_images`, and `extracted_features` will be created within `BC_MRI`.

4. Place the `requirements.txt` file in the main directory (`BC_MRI`) and install the required functions. Additional functions may need to be installed as needed using the `pip install` method.

5. Place the image data `Duke-Breast-Cancer-MRI.rar` in the `dataset` directory. If direct upload is not possible due to high volume, use Google Drive to upload it in the `dataset` directory as mentioned in the code.

6. After downloading data components (b, c, d, e) and converting them to CSV, put them in the `dataset` directory.

7. Start with the `Feature_Extraction.py` file to extract 12 combinations of radiomics features. After obtaining the 12 CSV data files, proceed to run the other `.py` files in this order:

   A) `Data_preprocessing.py`: Merge previous datasets, add some clinical features, and do some initial feature selection to get rid of redundant features before proceeding with machine learning algorithms.
   
   B) `Binary_Classifications_OvR_OvO.py`: Classify molecular subtypes of BC using two approaches:
   
      - One vs. the rest classifications (4 OvR classifications)
      - One vs. one classifications (6 OvO classifications)

   C) `Radiomics_Features_Saha_et.py`: Apply the same methodology for the extracted radiomics features by Saha et al. and make comparisons. [Saha et al. - British Journal of Cancer](https://www.nature.com/articles/s41416-018-0185-8)
