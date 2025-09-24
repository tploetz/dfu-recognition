from utils import *
import os
import cv2
import numpy as np
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert annotations to masks')
    parser.add_argument('--input_dir', '-i', type=str, default='./data/origin',
                        help='Input directory containing images and JSON annotations')
    parser.add_argument('--output_dir', '-o', type=str, default='./data/DFU',
                        help='Output directory for processed images and masks')
    
    args = parser.parse_args()
    
    root = args.input_dir
    output_dir = args.output_dir
    
    ducss_img_dir = os.path.join(output_dir, 'images')
    ducss_label_dir = os.path.join(output_dir, 'labels')
    statistic_txt = os.path.join(output_dir, 'statistic.txt')
    os.makedirs(ducss_img_dir, exist_ok=True)
    os.makedirs(ducss_label_dir, exist_ok=True)

    img_num = len(os.listdir(root))/2
    img_count = 0
    print(f"Total {img_num} images")

    views = ['dorsal', 'plantar', 'medial', 'lateral', 'toetips', 'heel']
    anno_type = ['ulcer', 'lesion', 'healed scar']

    sta_file = open(statistic_txt, 'w')

    for img_name in os.listdir(root):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_count += 1
            printProgressBar(img_count, img_num-1)

            img_path = os.path.join(root, img_name)
            json_path = img_path.replace('.jpg', '.json')
            foot_type = ''
            view = 6
            ulcer_num = 0
            lesion_num = 0
            
            boundary_points_list = []
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        print(Fore.RED +  "JSON file broken: " + img_name)
                        continue
                    labeled_area = data['shapes']
                    for shape in labeled_area:
                        label = shape['label']
                        if label == 'left' or label == 'right':
                            foot_type = label
                            continue
                        if label in anno_type and label != 'healed scar':
                            if shape['shape_type'] == 'circle':
                                print(img_name)

                            if label == 'ulcer':
                                ulcer_num += 1
                            elif label == 'lesion':
                                lesion_num += 1
                            
                            boundary_points_list.append({'points': shape['points'], 'type': shape['shape_type']})
                            continue

                        if label in views:
                            view = views.index(label)
            else:
                print(Fore.RED +  "JSON file not found: " + img_name)
                
            sta_file.write(f"{img_name} {foot_type} {view} {ulcer_num} {lesion_num}\n")

            image = cv2.imread(img_path)
            height, width, _ = image.shape
            mask = np.zeros((height, width), dtype=np.uint8)
            for boundary in boundary_points_list:
                points = np.array(boundary['points'], dtype=np.int32)
                if boundary['type'] == 'linestrip' or boundary['type'] == 'polygon':
                    cv2.fillPoly(mask, [points], 255)
                elif boundary['type'] == 'circle':
                    radius = int(np.sqrt(
                        (points[0][0] - points[1][0])**2 + 
                        (points[0][1] - points[1][1])**2
                    ))
                    cv2.circle(mask, (points[0][0], points[0][1]), radius, 255, -1)
                    
            mask_path = os.path.join(ducss_label_dir, img_name[:-4] + '.png')
            cv2.imwrite(mask_path, mask)
            shutil.copy(img_path, os.path.join(ducss_img_dir, img_name))

    print(img_count)
    sta_file.close()

if __name__ == "__main__":
    main()