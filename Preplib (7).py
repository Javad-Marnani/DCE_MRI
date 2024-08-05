import os
import cv2
import math
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def maskGenerator(patient_folder_path, rotate, _thr_remove_percent=0.2, cut1_threshold=40, cut2_threshold=400, cut1_threshold_shift=40, save_folder_name='masked_images', save=False, show=False):
    """
    Generates a mask for the breast MRI image

    Parameters:
    patient_folder_path (str): Path to the patient's folder containing pre/post folders.
    rotate (bool): If True, rotates the image by 180 degrees.
    _thr_remove_percent (float): The percentage of initial and final slices to discard.
    cut1_threshold (int): Threshold for the first cutting process.
    cut2_threshold (int): Threshold for the second cutting process.
    cut1_threshold_shift (int): Shift value for the cut1 threshold adjustment.
    save_folder_name (str): Name of the folder where the resulting images will be saved.
    save (bool): If True, saves the generated mask images.
    show (bool): If True, displays the generated mask image.

    Returns:
    ndarray: The processed image after all operations (masking, cutting, etc.).

    Description:
    The function processes medical images by reading slices, applying thresholds and other transformations,
    and potentially saving the output. It involves steps such as sorting the image slices, removing a percentage
    of slices from the ends, adding images from multiple posts, applying rotations and resizing, and finally
    performing image thresholding and cutting based on specified thresholds and points.
    """

    # Reading slices' names, and sorting them
    slices = list(os.listdir(os.path.join(patient_folder_path, 'pre')))
    # Filter to include only .jpg files
    slices = [s for s in slices if s.endswith('.jpg')]
    slices = sorted(slices, key=lambda x: int(x.split('.')[0]))
    # Number <1> in the diagram
    # Removing p percent from start and ending slices
    slices = remove_p_percent_from_ends(slices, _thr_remove_percent)

    # Getting image original size
    img_orig_size = cv2.imread(os.path.join(patient_folder_path, 'pre', slices[1])).shape[:-1]

    # Retrieving posts available for current patient
    patient_posts = get_folder_names_with_post(patient_folder_path)

    # We add corresponding slices together.
    post_pre_added_together = []
    for slice in slices:
        # Add up pre and all posts slices together
        img = cv2.imread(os.path.join(patient_folder_path, 'pre', slice))
        for post in patient_posts:
            img = cv2.add(img, cv2.imread(os.path.join(patient_folder_path, post, slice)))
        # Rotating if needed
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        # Resizing
        img = cv2.resize(img, (512, 512))
        # Adding to the list
        post_pre_added_together.append(img)

    # Add up all added slices together
    slice = addWithOpacity(post_pre_added_together)

    # Finding the mask
    mask = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
    mask = thresholding(mask, getBackgroundColor(mask))
    mask = intensifier(mask, 6)

    # Finding c1, c2, c3 points
    c1, c2, c3 = breastCutPoints(slice, cut1_threshold, cut2_threshold, cut1_threshold_shift)

    # Cutting the breast using the mask
    result = curve(mask, [(0, c2), (256, c1), (512, c2)], 1000, True)

    # Returning to original size
    result = cv2.resize(result, img_orig_size)

    # Saving
    if save:
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
        patient_number = patient_folder_path.split('/')[-1].strip('patient')
        cv2.imwrite(os.path.join(save_folder_name, patient_number + '.png'), result)

    # Show the result if the 'show' parameter is True
    if show:
        plt.imshow(result, cmap='gray')
        plt.title("Generated Mask")
        plt.show()

    return result

def edgeRemover(image, intensifier_value=6, thickness_value=10, color=None):
    image_ref = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = thresholding(image, getBackgroundColor(image))
    image = intensifier(image, intensifier_value)
    first_white_pixels = np.zeros(image.shape[1], dtype=int)
    for col in range(image.shape[1]):
        white_pixel_index = np.argmax(image[:, col])
        first_white_pixels[col] = white_pixel_index
    if color != None:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = (0, 0, 0)
    points = np.column_stack(
        (np.arange(len(first_white_pixels)), first_white_pixels))
    for i in range(1, len(points)):
        cv2.line(image_ref, tuple(points[i - 1]),
                 tuple(points[i]), color, thickness_value)
    if color == None:
        image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)
    return image_ref, points


def point_remover(image, points, thickness=10):
    for i in range(1, len(points)):
        cv2.line(image, tuple(points[i - 1]),
                 tuple(points[i]), (0, 0, 0), thickness)
    return image


def remove_p_percent_from_ends(lst, p):
    num_elements_to_remove = int(len(lst) * p)
    start_index = num_elements_to_remove
    end_index = len(lst) - num_elements_to_remove
    new_lst = lst[start_index:end_index]
    return new_lst


def normalize(img):
    img_float = np.float32(img)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    normalized_img = 255 * (img_float - min_val) / (max_val - min_val)
    normalized_img_uint8 = normalized_img.astype(np.uint8)
    return normalized_img_uint8


def rowPlot(size=(20, 10), image_list=None):
    plt.figure(figsize=size)
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i+1)
        plt.imshow(image_list[i])
        plt.title(i+1)
        plt.axis('off')
    plt.show()


def addWithOpacity(image_list):
    opacity = 1 / len(image_list)
    for i in range(len(image_list)):
        image_list[i] = np.clip(image_list[i], 0, 255)
        image_list[i] = image_list[i].astype(np.float32)
        image_list[i] = image_list[i] * opacity
    added_images = sum(image_list)
    layered = np.clip(added_images, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(layered, cv2.COLOR_BGR2RGB)
    result = normalize(result)
    return result


def curve(image, points, num_points, remove):
    points = np.array(points, dtype=np.int32)
    curve = np.poly1d(np.polyfit(points[:, 0], points[:, 1], 2))
    x_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), num_points)
    y_vals = curve(x_vals).astype(np.int32)
    curve_points = np.column_stack((x_vals, y_vals)).astype(np.int32)
    for i in range(1, len(curve_points)):
        pt1 = (curve_points[i - 1][0], curve_points[i - 1][1])
        pt2 = (curve_points[i][0], curve_points[i][1])
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    if remove:
        for pnt in curve_points:
            cv2.line(image, pnt, (pnt[0], 512), (0, 0, 0), 5)
    return image


def plotAll(lst, num, save):
    nrows, ncols = math.ceil(num/6), 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 100))
    for i, ax in enumerate(axes.flat):
        if i < len(lst):
            ax.imshow(lst[i])
            ax.set_title(str(i+1), fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    if save:
        plt.savefig('images_grid_high_res.png', dpi=300)
    plt.tight_layout()
    plt.show()


def getBackgroundColor(image):
    A1 = image[0:30, :]
    A2 = image[0:100, 0:30]
    A3 = image[0:100:, -30:]
    A1 = cv2.mean(A1)
    A1 = A1[0]
    A2 = cv2.mean(A2)
    A2 = A2[0]
    A3 = cv2.mean(A3)
    A3 = A3[0]
    return (A1+A2+A3)/3


def thresholding(image, thr):
    close_to_average_threshold = 40
    close_to_white_threshold = 255
    custom_thresholded_image = np.where(
        (image > thr - close_to_average_threshold) & (image < thr + close_to_average_threshold), 0, image)
    image = np.where(image >= close_to_white_threshold,
                     255, custom_thresholded_image)
    return image


def intensifier(image, cnt):
    image = cv2.add(image, image)
    for i in range(cnt - 2):
        image = cv2.add(image, image)
    return image


def breastCut(image, cut1_threshold=40, cut2_threshold=400, cut1_threshold_shift=40):
    c1 = -1
    c2 = -1

    for r in range(1, 256):
        b = list(image[256-r:256-r+1, 110:340][0])
        first_255 = b.index(255)
        last_255 = len(b) - b[::-1].index(255) - 1
        middle_zeroes = b[first_255:last_255].count(0)
        if middle_zeroes > cut1_threshold:
            c1 = (256 - r + cut1_threshold_shift)
            break
    for r in range(1, 256):
        b = list(image[256+r:256+r+1, :][0])
        first_255 = b.index(0)
        last_255 = len(b) - b[::-1].index(0) - 1
        middle_zeroes = b[first_255:last_255].count(255)
        if middle_zeroes > cut2_threshold:
            c2 = (256 + r)
            break

    if c1 > c2:
        c2 = c1 + cut1_threshold_shift

    return c1, c2


def borderCrop(image, border_crop_thr):
    image[:border_crop_thr, :] = 0
    image[-border_crop_thr:, :] = 0
    image[:, -border_crop_thr:] = 0
    image[:, :border_crop_thr] = 0
    return image


def topBreastCut(image):
    for r in range(1, 512):
        n = image[:r, :]
        if 255 in n:
            image[:r, :] = 0
            break
    return r


def breastCutPoints(image, cut1_threshold=40, cut2_threshold=400, cut1_threshold_shift=40):
    # Converting to the grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding the image
    image_thr = thresholding(image, getBackgroundColor(image))
    # Intensify the image
    image_ins = intensifier(image_thr, 6)
    # Finding the cutting the breast points
    c1, c2 = breastCut(image_ins, cut1_threshold=cut1_threshold,
                       cut2_threshold=cut2_threshold, cut1_threshold_shift=cut1_threshold_shift)
    # Cutting the breast
    image = curve(image, [(0, c2), (256, c1), (512, c2)], 1000, True)
    # Bordercrop intensified images
    image_ins = borderCrop(image_ins, 10)
    # Cutting top breast
    c3 = topBreastCut(image_ins)
    return c1, c2, c3


def get_folder_names_with_post(path):
    folder_names = [name for name in os.listdir(path)
                    if os.path.isdir(os.path.join(path, name)) and 'post' in name.lower()]
    return folder_names


def min_max_normalization(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array
