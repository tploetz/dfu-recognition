import os 
import cv2
import shutil
import os
import json
from colorama import Fore
import numpy as np
import sys
from PIL import Image

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



def json2Mask(input_dir, output_dir, mask_type):

    output_image_dir = os.path.join(output_dir, 'images')
    output_mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    img_num = len(os.listdir(input_dir))/2
    img_count = 0

    for img in os.listdir(input_dir):
        if img.endswith('.jpg'):
            img_count += 1
            print("Process progress: {}%: ".format(np.ceil(img_count/img_num*100)), "▋" * (int(img_count/img_num*100 )// 2), end="\r")
            sys.stdout.flush()

            img_path = os.path.join(input_dir, img)
            json_path = img_path.replace('.jpg', '.json')

            boundary_points_list = []
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        print(Fore.RED +  "JSON file broken: " + img)
                        continue
                    labeled_area = data['shapes']
                    for shape in labeled_area:
                        label = shape['label'] #left , right, ulcer, lesion, 
                        boundary_points = shape['points']
                        if label in mask_type:
                            boundary_points_list.append(boundary_points)      
            else:
                print(Fore.RED +  "JSON file not found: " + img)

            image = cv2.imread(img_path)
    
            height, width, _ = image.shape
            mask = np.zeros((height, width), dtype=np.uint8)
            for boundary in boundary_points_list:
                points = np.array(boundary, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

            mask_path = os.path.join(output_mask_dir, img.replace('.jpg', '.png'))
            cv2.imwrite(mask_path, mask)
            shutil.copy(img_path, os.path.join(output_image_dir, img))


def resize_and_pad(img, size=(512, 512), pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  

    # computing scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, 
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img







def check_mask(mask_dir):
    all_images = 0
    have_mask = 0
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png'):
            all_images += 1
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = cv2.imread(mask_path)
            notBlank = mask.any()
            if notBlank:
                have_mask += 1
    print(f'Number of images: {all_images}')
    print(f'Number of images with masks: {have_mask}')
    print(f'Number of images without masks: {all_images - have_mask}')

def remove_blank_mask(mask_dir, img_dir):
    all_images = 0
    have_mask = 0
    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png'):
            all_images += 1
            mask_path = os.path.join(mask_dir, mask_filename)
            img_path = os.path.join(img_dir, mask_filename[:-4]+'.jpg')
            mask = cv2.imread(mask_path)
            notBlank = mask.any()
            if notBlank:
                have_mask += 1
            else:
                os.remove(mask_path)
                os.remove(img_path)
    print(f'Number of images: {all_images}')
    print(f'Number of images with masks: {have_mask}')


def add_mask2img(img_path, mask_path):
    # print(f'Processing {img_path}...')
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img_with_mask1 = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    return img_with_mask1
    # Save the ulcer only mask
    # cv2.imshow('img_with_mask', img_with_mask1)
    # cv2.waitKey(0)

def combineDir(input_dir_list, dest_dir, show_exist=False):
    os.makedirs(dest_dir, exist_ok=True)
    for input_dir in input_dir_list:
        count = 0
        for img in os.listdir(input_dir):
            if img.endswith('.png') or img.endswith('.jpg'):
                img_path = os.path.join(input_dir, img)
                
                if img in os.listdir(dest_dir):
                    if show_exist:
                        print(f'File {img} already exists')
                    continue
                new_img_path = os.path.join(dest_dir, img)
                shutil.copy(img_path, new_img_path)
                count += 1
        print(f'Copy {count} images from {input_dir} to {dest_dir}')




def mask_augment(mask_dir):
    for mask in os.listdir(mask_dir):
        if mask.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

            cv2.imwrite(mask_path, mask)