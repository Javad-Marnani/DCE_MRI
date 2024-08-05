##########################################################################################
#####################################   IMPORTS    #######################################
##########################################################################################
import os
import cv2 as cv
import math
import subprocess
import itertools
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
import re
import pickle
import subprocess
import operator
import pickle
import matplotlib.pyplot as plt
from six_coords_predictor import *
from Preplib import *

# Print the current working directory
current_directory = os.getcwd()
print(f"Current Working Directory: {current_directory}")

# extract_zip("/users/fs2/jaghadavoodm/yolo/patient_images200.zip","/users/fs2/jaghadavoodm/yolo/All922Patients")

# extract_zip("/users/fs2/jaghadavoodm/yolo/patient_images400.zip","/users/fs2/jaghadavoodm/yolo/All922Patients")

# extract_zip("/users/fs2/jaghadavoodm/yolo/patient_images600.zip","/users/fs2/jaghadavoodm/yolo/All922Patients")

# extract_zip("/users/fs2/jaghadavoodm/yolo/patient_images800.zip","/users/fs2/jaghadavoodm/yolo/All922Patients")

# extract_zip("/users/fs2/jaghadavoodm/yolo/patient_images922.zip","/users/fs2/jaghadavoodm/yolo/All922Patients")

"""
Plot one image from each patient's sequence to identify and correct any images that are oriented downwards. This helps ensure all images are uniform and facing upwards for consistent analysis.
"""
all_images_directory= '/users/fs2/jaghadavoodm/yolo/All922Patients'
#show_patient_images(all_images_directory, 'pre', image_number=50, num_columns=10, batch_size=100)

# This list of patients to rotate is currently hard-coded, but we may consider making it configurable in the future to automatically ensure all images are uniformly oriented upwards.
images_to_rotate = [2,3,4,6,9,10,11,12,13,16,20,21,24,25,29,30,31,32,33,34,35,36,38,41,43,45,47,49,50,51,53,54,55,56,58,59,60,
                    61,62,64,65,66,68,69,70,71,73,77,78,79,80,83,87,88,89,90,93,99,102,105,107,108,109,110,111,113,114,115,117,
                    118,122,123,129,130,133,134,135,136,138,145,146,147,148,150,151,152,154,155,159,160,161,162,163,167,170,172,
                    173,174,175,178,179,182,184,185,186,187,190,191,193,194,199,203,205,206,207,208,212,213,216,217,218,219,220,
                    222,225,226,228,229,230,232,233,234,235,236,238,239,240,241,242,243,244,245,247,248,249,251,252,254,255,257,
                    258,259,260,261,262,263,264,267,269,272,273,274,276,277,278,279,280,281,283,284,285,286,287,289,291,292,295,
                    296,297,298,299,300,301,302,303,305,306,308,309,310,311,312,314,315,317,318,319,320,322,325,326,328,329,330,
                    331,332,333,334,336,337,338,339,340,343,345,346,347,348,349,350,352,354,355,357,358,359,360,361,362,364,365,
                    366,367,369,372,373,374,377,378,380,381,382,383,387,388,389,390,391,392,393,395,396,397,399,400,401,402,403,
                    404,406,407,408,411,412,413,414,415,416,418,419,412,422,423,424,425,426,427,430,432,433,434,435,438,439,440,
                    441,442,443,444,445,447,448,449,450,451,452,455,458,459,460,461,462,463,465,466,467,468,469,470,471,472,473,
                    475,476,480,481,483,486,488,490,491,492,494,495,496,497,498,499,501,503,504,505,507,508,511,513,514,515,517,
                    518,519,520,521,522,523,524,525,528,530,531,532,535,536,537,539,540,542,543,547,548,549,550,551,552,553,555,
                    556,557,558,560,561,563,565,566,567,571,572,573,574,575,576,579,581,583,584,585,586,588,590,592,593,594,597,
                    599,602,603,604,606,608,609,610,611,615,616,617,618,619,620,621,623,624,625,626,627,630,632,633,634,635,636,
                    637,638,639,640,642,643,646,648,649,650,652,653,654,655,657,658,660,661,663,664,665,666,667,668,669,670,672,
                    673,674,675,676,677,678,682,684,685,686,687,688,689,691,692,694,699,701,703,704,705,706,708,709,712,713,715,
                    716,719,720,721,722,723,726,730,732,733,734,735,736,737,739,740,742,743,744,745,746,747,748,750,751,752,753,
                    755,757,758,760,761,762,763,765,766,767,768,769,770,772,774,775,777,779,781,784,785,786,787,789,790,792,793,
                    794,795,796,798,800,803,804,806,807,809,810,811,812,813,814,815,816,817,820,821,822,823,825,826,827,828,829,
                    831,832,833,834,835,836,837,838,839,840,841,842,843,845,846,848,852,853,855,856,858,859,860,861,862,863,865,
                    867,868,869,870,871,872,873,874,875,876,877,878,879,880,882,884,885,887,888,889,891,892,894,895,896,897,898,
                    899,902,903,905,906,908,909,910,911,912,913,914,915,916,917,918,919,921,922
                   ]

num_patients = 922

# Generating masks to segment the breasts and exclude the background and chest area
for p in tqdm(range(1, num_patients+1),desc="Generating masks to segment the breasts, distinguishing them from the background and chest areas."):
    if p == 468:  # Skipping patient 468
        continue

    # Constructing the path to each patient's folder directly
    patient_folder_path = f'{all_images_directory}/patient{p}'

    # Determine whether to rotate images based on the presence in the list
    rotate = True if p in images_to_rotate else False

    # Generating masks with specified parameters
    mask = maskGenerator(patient_folder_path,
                         rotate,
                         _thr_remove_percent=0.2,
                         cut1_threshold=40,
                         cut2_threshold=400,
                         cut1_threshold_shift=60,
                         save_folder_name='masked_images', 
                         save=True,
                         show=False)

masks_path= '/users/fs2/jaghadavoodm/yolo/masked_images'
annotations_path='/users/fs2/jaghadavoodm/yolo/Annotation_Boxes.csv'
sizes = get_image_sizes(masks_path)
#print(sizes)

patient_slice_counts = count_patient_slices(all_images_directory,num_patients=922, trim_percentage=0.40)
#print(patient_slice_counts)

cube_coordinates=cube_coordinates(annotations_path,num_patients=922)


adjusted_z_coordinates=adjust_z_coordinates(cube_coordinates, patient_slice_counts, percentage=0.40)


scaled_cube_coordinates=scale_bounding_box_for_rotation(adjusted_z_coordinates, images_to_rotate, sizes)
#print(scaled_cube_coordinates)

masks=plot_masked_images_with_bbox(masks_path, scaled_cube_coordinates, num_columns=10,show=True)

patients_to_exclude=[25,56,83,111,130,171,182,194,205,213,215,250,301,421,450,468,473,489,512,523,577,600,605,648,655,664,
                    671,680,691,704,717,719,793,808,858,910]
print("Number of patients with bounding boxes completely or partly outside the breast areas:",len(patients_to_exclude))

print("Total number of patients after excluding those with bounding boxes outside the breast areas:",num_patients-len(patients_to_exclude))
# Constructing RGB images from pre-contrast and post-contrast sequences 1 and 2
rgb_images = data_loader(
    base_path=all_images_directory, 
    patient_slice_counts=patient_slice_counts, 
    scaled_cube_coordinates=scaled_cube_coordinates, 
    patients_to_rotate=images_to_rotate, 
    num_patients=num_patients, 
    patients_to_exclude=patients_to_exclude, 
    seq_type1='pre', 
    seq_type2='post1', 
    seq_type3='post2'
)

for index, result in enumerate(rgb_images, start=1):
    print(f"{index}. Processed patient {result['patient_number']}: {len(result['rgb_volume'])} rgb images")

print("rgb_images_len",len(rgb_images))

#Applying masks over the slices to segment the breasts
masked_results = apply_mask_to_slices(rgb_images, masks)

# To verify the masking operation, you can print some details for a subset of patients
for result in masked_results[:5]:  # Print information for the first 5 patients, as an example
    patient_number = result['patient_number']
    num_images = len(result['rgb_volume'])
    print(f"Processed patient {patient_number}: {num_images} RGB images with applied mask")

# Resizing the slices along the three axes: x, y, and z
resized_volumes=resize_rgb_images(masked_results, new_size=(100,256,256, 3))
#print(resized_volumes[0])
print("Save the resized volumes as a list of dictionaries in a pickle file to avoid re-running previous sessions and save time.")
# Save the list of dictionaries in pickle format to avoid rerunning previous sections and to save time.
# # Saving the data with pickle
# with open('resized_volumes.pkl', 'wb') as f:
#     pickle.dump(resized_volumes, f)

# # Load the saved resized data from a pickle file, expecting a list of dictionaries
# with open('resized_volumes.pkl', 'rb') as f:
#     resized_volumes = pickle.load(f)
#print(resized_volumes[0])

patient_to_plot = 500  # Example patient number
plot_resized_patient_volumes(resized_volumes, patient_to_plot, num_columns=5)
# Split the data into train and test sets using a systematic sampling method to ensure consistent splits across different runs, starting from a fixed start point.
test_percentage = 0.2
start_point = 2  # Adjust this to change the starting point for test patient selection
train_data, test_data = split_patient_data(resized_volumes, test_percentage, start_point)
#print(test_data[0])

print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

print("==================================================")
print("======== Setting Up YOLOv5 Object Detection =======")
print("==================================================")

# Save the training data images and labels in the specified directory for YOLOv5 training
save_images_and_labels(train_data, '/users/fs2/jaghadavoodm/yolo/all886_train')
# Save the testing data images and labels in the specified directory for YOLOv5 testing
save_images_and_labels(test_data, '/users/fs2/jaghadavoodm/yolo/all886_test')

# Define the content for the dataset configuration file (dataset.yaml)
# This configuration specifies the paths to training and validation datasets, the number of classes (nc),
# and the class names for use with YOLOv5.
dataset_yaml_content = """
train: /users/fs2/jaghadavoodm/yolo/all886_train/train/images
val: /users/fs2/jaghadavoodm/yolo/all886_test/test/images

nc: 1
names: ['tumor']
"""

# Write the configuration settings to the 'all886.yaml' file for YOLOv5 dataset setup
with open('/users/fs2/jaghadavoodm/yolo/all886.yaml', 'w') as file:
    file.write(dataset_yaml_content)
# Define hyperparameters for YOLOv5 training in YAML format. This configuration adjusts learning rates, 
# momentum settings, loss gains, and augmentation parameters to optimize model performance.
hyp_yaml_content = """
lr0: 0.00258  # initial learning rate
lrf: 0.17  # final learning rate factor
momentum: 0.779  # SGD momentum
weight_decay: 0.00058  # weight decay
warmup_epochs: 1.33  # warmup epochs
warmup_momentum: 0.86  # warmup initial momentum
warmup_bias_lr: 0.0711  # warmup initial bias lr
box: 0.0539  # box loss gain
cls: 0.299  # class loss gain (might be less relevant if focusing on object detection only)
cls_pw: 0.825  # class BCELoss positive weight (as above)
obj: 0.632  # object loss gain
obj_pw: 1.0  # object BCELoss positive weight
iou_t: 0.2  # IoU training threshold
anchor_t: 3.44  # anchor threshold
fl_gamma: 0.0  # focal loss gamma
hsv_h: 0.0  # hue variation (not relevant for grayscale images)
hsv_s: 0.0  # saturation variation (not relevant for grayscale images)
hsv_v: 0.1  # slight brightness variation might be okay
degrees: 5.0  # small rotations might be okay
translate: 0.1  # slight translations can simulate patient movement
scale: 0.05  # minor scaling to account for size variations
shear: 5.0  # small shear might be okay
perspective: 0.0  # perspective changes are likely not relevant
flipud: 0.0  # vertical flips are less common, turn off unless specifically needed
fliplr: 0.5  # horizontal flips could simulate mirror images
mosaic: 0.0  # turn off for medical imaging
mixup: 0.0  # turn off for medical imaging
copy_paste: 0.0  # turn off for medical imaging
"""
# Write the hyperparameter settings to 'hyp.custom_.yaml' for customizing YOLOv5 training configurations.
with open('/users/fs2/jaghadavoodm/yolo/hyp.custom_.yaml', 'w') as file:
    file.write(hyp_yaml_content)


print(">>> =============================== <<<")
print(">>> Starting YOLOv5 Training Session <<<")
print(">>> =============================== <<<")


"""# Training

YOLOv5 offers several pre-trained weight options, each corresponding to a different model size and capacity. The main variants include:

- **YOLOv5n.pt** - The "nano" model, designed for ultra-fast inference and efficiency, suitable for edge devices with very limited computational resources.
- **YOLOv5s.pt** - The "small" model, designed for speed and efficiency, suitable for applications where computational resources are limited.
- **YOLOv5m.pt** - The "medium" model, offering a balance between speed and accuracy.
- **YOLOv5l.pt** - The "large" model, which provides higher accuracy at the cost of increased computational requirements.
- **YOLOv5x.pt** - The "extra large" model, the largest and most accurate, but also the most demanding in terms of computational resources.
"""


subprocess.run([
    'python', '/users/fs2/jaghadavoodm/yolo/yolov5/train.py',
    '--img', '256',
    '--batch', '128',
    '--epochs', '20',
    '--data', '/users/fs2/jaghadavoodm/yolo/all886.yaml',
    '--weights', 'yolov5x.pt',
    '--cache',
    '--hyp', '/users/fs2/jaghadavoodm/yolo/hyp.custom_.yaml',
    '--device', '1'
])

print("Note: For accurate loss function plotting, ensure the experiment folder is up-to-date with the results from the latest training session (e.g., use 'exp16' instead of 'exp15').")
# Define the path to the results CSV
results_csv_path = '/users/fs2/jaghadavoodm/yolo/yolov5/runs/train/exp103/results.csv'

# Call the function and store the path of the saved plot
saved_plot_path = plot_loss(results_csv_path)

# Print the path along with a message
print(f"This is the path of the saved loss function plot: {saved_plot_path}")

print(">>> =============================== <<<")
print(">>> Starting YOLOv5 Prediction Session on Test Data <<<")
print(">>> =============================== <<<")

print("Note: Before initiating predictions on test data, ensure the experiment folder reflects the most recent training results (e.g., use 'exp16' instead of 'exp15').")

"""# Prediction (Test)"""
detect_command = [
    'python', '/users/fs2/jaghadavoodm/yolo/yolov5/detect.py',
    '--weights', '/users/fs2/jaghadavoodm/yolo/yolov5/runs/train/exp103/weights/best.pt',
    '--img', '256',
    '--conf', '0.01',
    '--source', '/users/fs2/jaghadavoodm/yolo/all886_test/test/images',
    '--save-txt',
    '--save-conf',
    '--device', '0,1'
]

# Redirect output to a file
with open('temp_output.txt', 'w') as output_file:
    subprocess.run(detect_command, stdout=output_file, stderr=subprocess.STDOUT)

# Optionally, capture and print the last 10 lines of the output
with open('temp_output.txt', 'r') as output_file:
    lines = output_file.readlines()
    last_lines = lines[-10:]  # Adjust the number '-10' as necessary
    print(''.join(last_lines))

print("Note: Before processing predictions on test data, ensure the experiment folder contains the most recent results (e.g., use 'exp16' instead of 'exp15').")

# # #IMPORTANT: exp may need to be updated

# Correct the paths based on your setup
test_images_dir = '/users/fs2/jaghadavoodm/yolo/all886_test/test/images'
test_labels_dir = '/users/fs2/jaghadavoodm/yolo/all886_test/test/labels'  # Ground truth labels directory
pred_labels_dir = '/users/fs2/jaghadavoodm/yolo/yolov5/runs/detect/exp138/labels'  # Predicted labels directory
#IMPORTANT: exp may need to be updated
# List all label files in the predicted labels directory
pred_label_files = [f for f in os.listdir(pred_labels_dir) if f.endswith('.txt')]

# Count the number of label files
num_pred_labels = len(pred_label_files)
print("Number of predicted label files:", num_pred_labels)

# Optionally, print the names of the label files
# for file_name in pred_label_files:
#     print(file_name)

# Define your directories
# Create a new directory path for processed labels by replacing 'labels' with 'processed_labels' in the pred_labels_dir
output_dir = pred_labels_dir.replace('labels', 'processed_labels')
# Print the new directory path to confirm it's correct
print("Output directory for processed labels:", output_dir)
os.makedirs(output_dir, exist_ok=True)

# Process detections
process_detections(pred_labels_dir, output_dir)
processed_label_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]

# Count the number of label files
num_processed_labels = len(processed_label_files)
print("Number of processed label files:", num_processed_labels)

# Optionally, print the names of the label files
# for file_name in processed_label_files:
#     print(file_name)

# List label files ending with '.txt'
processed_label_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
# Sort the list of filenames by patient number and then by slice number
sorted_label_files = sorted(processed_label_files, key=get_patient_and_slice_number)

# Count the number of label files
num_processed_labels = len(sorted_label_files)
print("Number of processed label files:", num_processed_labels)

# Print the names of the sorted label files along with their contents
# for file_name in sorted_label_files:
#     file_path = os.path.join(output_dir, file_name)
#     with open(file_path, 'r') as file:
#         contents = file.read().strip()  # Read and strip unnecessary whitespace
#     print(file_name, ":", contents)


# List label files ending with '.txt'
processed_label_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]


# Sort the list of filenames by patient number and then by slice number
sorted_label_files = sorted(processed_label_files, key=get_patient_and_slice_number)

# Prepare a list to hold the confidence results as dictionaries
confidence_results_test = []

# Collect data in the required format
for file_name in sorted_label_files:
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'r') as file:
        contents = file.read().strip()
    # Split to get the individual elements from the file contents
    elements = contents.split()
    # The last element is the confidence
    confidence = float(elements[-1])
    # Patient number and slice number from filename
    patient_number, slice_number = get_patient_and_slice_number(file_name)
    # Append to confidence_results list
    confidence_results_test.append({'patient_number': patient_number, 'slice_number': slice_number, 'confidence': confidence})

# Print confidence results
# for result in confidence_results_test:
#     print(result)

# print(confidence_results_test[0])
test_patients=[item['patient_number'] for item in test_data]
print("Number of patients in test:",len(test_patients))


"""# Visualization (Test)"""
patient_results_test = []
for patient in test_patients:
    print(f"This is patient {patient} in test data")
    # Calling the updated function directly with the patient_number and capturing the returned results
    patient_results = visualize_patient_data(test_data, patient, pred_labels_dir, num_columns=6, use_processed_predictions=True,show=False)
    if patient_results:  # If there are results, extend the all_patient_results list with them
        patient_results_test.extend(patient_results)


#print(len(patient_results_test))
#print(patient_results_test)

overlap_stats_per_patient = calculate_overlap_stats_with_true_false(patient_results_test)

# Printing the results in a readable format
for patient, stats in overlap_stats_per_patient.items():
    print(f"Patient {patient}: True Overlaps: {stats['true_overlaps']}, "
          f"False Overlaps: {stats['false_overlaps']}, "
          f"Non-Overlaps: {stats['non_overlaps']}, "
          f"No Prediction: {stats['no_prediction']}")

slice_number=100
# Calculating the percentages for each category
total_patients = len(overlap_stats_per_patient)
sum_true_overlaps = sum(stats["true_overlaps"] for stats in overlap_stats_per_patient.values())
sum_false_overlaps = sum(stats["false_overlaps"] for stats in overlap_stats_per_patient.values())
sum_non_overlaps = sum(stats["non_overlaps"] for stats in overlap_stats_per_patient.values())
sum_no_prediction = sum(stats["no_prediction"] for stats in overlap_stats_per_patient.values())
avg_true_overlaps = (sum_true_overlaps / total_patients) / slice_number
avg_false_overlaps = (sum_false_overlaps / total_patients) / slice_number
avg_non_overlaps = (sum_non_overlaps / total_patients) / slice_number
avg_no_prediction = (sum_no_prediction / total_patients) / slice_number

# Printing the calculated percentages for each category
print("True Overlaps Percentage: {:.2%}".format(avg_true_overlaps))
print("False Overlaps Percentage: {:.2%}".format(avg_false_overlaps))
print("Non Overlaps Percentage: {:.2%}".format(avg_non_overlaps))
print("No Prediction Percentage: {:.2%}".format(avg_no_prediction))

similar_bboxes_test=find_similar_bboxes_center_distance(patient_results_test,confidence_results_test, max_distance=14, min_group_size=3, comparison_range=5)
similar_bboxes_test

visualize_aggregated_images_with_bboxes(test_data, similar_bboxes_test)
visualize_aggregated_images_with_highest_avg_conf(test_data, similar_bboxes_test,confidence_threshold=0.05)

print("""
End of Execution
----------------
✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦
❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖ ❖
♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦ ♦
""")


