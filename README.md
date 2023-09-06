# DCE_MRI
# Breast Cancer Subtype Prediction Using DCE-MRI and Machine Learning

Below are the instructions, as outlined in the readme file, for accessing and utilizing the data to predict four molecular subtypes of breast cancer using DCE-MRI and machine learning algorithms.

## Dataset

The dataset can be accessed through this link: [Dataset Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). Download the following data components from the Data Access tab:

- Images (DICOM, 368.4 GB)
- File Path mapping tables (XLSX, 49.6 MB)
- Clinical and Other Features (XLSX, 582 kB)
- Annotation Boxes (XLSX, 49 kB)
- Imaging features (XLSX, 6.44 MB)

## Instructions

1. To filter and download the images, use the NBIA Data Retriever tool. You can find more information about downloading TCIA images using NBIA Data Retriever at this link: [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Select the "Classic Directory Name" option in NBIA Data Retriever.

2. Convert all the downloaded image data to RAR format to prepare for uploading to the desired directory.

3. Set up the following folders to organize your data:

   - Create a "DATA" folder to store all the required raw datasets.
   - Within the main directory, create a "Resized_Images" folder to store the resized images in PNG format.
   - Within the main directory, create an "Extracted Features" folder to store the extracted radiomics features (13 CSV files).
  
4. After extracting the RAR file(s), navigate to the "Balanced_Data" folder within the DATA folder. Create a folder called "Duke-Breast-Cancer-MRI" and place it alongside the "metadata.csv" file.

5. Convert all the Excel files (components 2 to 5) to CSV format.

6. Start with the "Feature Extraction" folder to extract 12 combinations of radiomics features.

7. Navigate to the "Classification" folder to run the codes for classification. Make any necessary modifications to paths and directories.

8. Ensure that you have installed the required libraries, such as Pyfeats and Mahotas. You may need to upgrade or downgrade your numpy library to install these libraries.
