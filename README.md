# Machine Learning-based Prediction of Molecular Subtypes of Breast Cancer using DCE-MRI

This README file contains instructions on how to access and utilize the data for predicting four molecular subtypes of breast cancer (Luminal A, Luminal B, HER2+, Triple Negative) using DCE-MRI and machine learning algorithms. Additionally, it provides guidance on properly running the codes in the intended order.

## Dataset

The dataset can be accessed through this link: [Dataset Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903). Download the following data components from the Data Access tab:

a) Images (DICOM, 368.4 GB)

b) File Path mapping tables (XLSX, 49.6 MB)

c) Clinical and Other Features (XLSX, 582 kB)

d) Annotation Boxes (XLSX, 49 kB)

e) Imaging features (XLSX, 6.44 MB)


## Instructions


1. All required directories will be created automatically if they do not exist. The main directory is `BC_MRI` created in the current directory. Four other directories including `xlsx_csv_files`, `dataset`, `resized_images`, and `extracted_features` will be created within `BC_MRI`.

2. Place the `requirements.txt` file in the main directory (`BC_MRI`) and install the required functions. Additional functions may need to be installed as needed using the `pip install` method.

3. After downloading the data components (b, c, d, e), place them in the `xlsx_csv_files` directory. The components will undergo conversion into CSV files and will be automatically saved in the same `xlsx_csv_files` directory.

4. Use the NBIA Data Retriever tool to filter and download the images. More information about downloading TCIA images using NBIA Data Retriever can be found at this link: [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). To run the codes for different samples, it is ideal to download all 922 patient folders. However, due to storage limitations, you can choose a subset of 200 patient folders or fewer. This subset can be generated using the `random_sample_for_each_cancer_type` function in the `Feature_Extraction.py` file. In the provided code, we have included a subset of 8 patients, with two patients for each cancer type (N0=N1=N2=N3=2). This subset of the dataset is specifically intended for testing purposes, ensuring that the codes can be executed even on a local machine. If you require a larger sample, you can modify the function arguments of `random_sample_for_each_cancer_type` accordingly. It is important to note that the size of the sample you choose may have an impact on the accuracy and reliability of the results. For more meaningful outcomes, it is recommended to work with a representative subset of patient folders. For example, you can consider setting N0=N1=N2=N3=50 to ensure a more comprehensive analysis. However, please keep in mind that by increasing the sample size, it becomes necessary to have sufficient storage capacity and computational capability due to the high volume of the data. Ensure that your storage resources can accommodate the increased number of patient folders and that your computational resources can handle the processing requirements. Additionally, working with a larger sample may require more time for data preprocessing, feature extraction, and analysis. It is advisable to assess your available resources and allocate sufficient time for the execution of the code when working with a larger sample size.

5. To ensure proper downloading of the images, please select the "Classic Directory Name" option in NBIA Data Retriever. By default, the option is set to "Descriptive Directory Name," so make sure to change it to "Classic Directory Name" before initiating the download. It is crucial to verify that all the images have been successfully downloaded. In cases where the sample size is larger, you may encounter errors during the process. If this occurs, simply complete the ongoing process and then proceed to redownload the remaining images. NBIA Data Retriever can utilize the previously set directory to download the remaining data.

6. Once you have downloaded the image data, ensure that you place the `Duke-Breast-Cancer-MRI` folder in the `dataset` directory. In case you encounter challenges uploading the folder due to its large size, consider compressing it into a RAR file and then extracting it. This approach can be particularly useful when running the codes on a server instead of locally.

7. Start with the `Feature_Extraction.py` file to extract 12 combinations of radiomics features. After obtaining the 12 CSV data files, proceed to run the other `.py` files in this order:

   A) `Data_preprocessing.py`: Merge previous datasets, add some clinical features, and do some initial feature selection to get rid of redundant features before proceeding with machine learning algorithms.
   
   B) `Binary Classifications_OvR_OvO.py`: Classify molecular subtypes of BC using two approaches:
   
      - One vs. the rest classifications (4 OvR classifications)
      - One vs. one classifications (6 OvO classifications)

   C) `Radiomics_Features_Saha_et.py`: Apply the same methodology for the extracted radiomics features by Saha et al. and make comparisons. [Saha et al. - British Journal of Cancer](https://www.nature.com/articles/s41416-018-0185-8)

8. Alternatively, you can run `main.py` instead of executing the four separate code files to achieve the same results. Although `main.py` offers simplicity in terms of execution, you may lose insight into the code flow and the underlying processes happening within different functions.

9. In the current version, we are running the codes with a small sample of 8 patients for testing purposes, ensuring they can be executed locally without errors. To avoid potential issues, we have commented out some lines at the end of the `Binary Classifications_OvR_OvO.py` file. This is because the sample size of 8 is too small for meaningful classification purposes and may lead to errors with certain classifiers. When working with a larger sample size, feel free to uncomment these lines. Additionally, you may need to modify the arguments of the functions throughout the codes to suit your specific needs.

10. Please be aware that if you plan to change the samples, it is crucial to delete the existing contents of the `dataset`, `resized_images`, and `extracted_features` directories. These directories will not automatically update their contents with each run, and if any files already exist within them, the results may not accurately reflect the new patient's samples. To ensure precise results, it is necessary to manually remove the previous contents before proceeding with new samples. 
