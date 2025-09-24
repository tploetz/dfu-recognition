#ducss statistic
# sta_file = open(statistic_txt, 'w')
import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='DUCSS statistics analysis')
    parser.add_argument('--inputdir', '-i', type=str, default='./data/DFU',
                        help='Output directory path (default: ./data/DFU)')
    
    args = parser.parse_args()
    
    input_dir = args.inputdir
    statistic_txt = os.path.join(input_dir, 'statistic.txt')
    
    sta_file = open(statistic_txt, 'r')

    ulcer_view = [0 for i in range(7)]
    lesion_view = [0 for i in range(7)]

    part_ulcer = []
    part_lesion = []
    part_concern = []
    ulcer_image_num = 0
    lesion_image_num = 0
    concern_image_num = 0
    ulcer_num = 0
    lesion_num = 0

    for sta_data in sta_file.readlines():
        sta_data = sta_data.strip().split(' ')
        # img_name = sta_data[0]
        participant_id = int(sta_data[0].split('_')[1])

        foot_type = sta_data[-4]
        view = int(sta_data[-3])
        ulcer_num = int(sta_data[-2])
        lesion_num = int(sta_data[-1])

        ulcer_view[view] += ulcer_num
        lesion_view[view] += lesion_num
        if ulcer_num > 0:
            ulcer_image_num += 1
            if participant_id not in part_ulcer:
                part_ulcer.append(participant_id)
           
        if lesion_num > 0:
            lesion_image_num += 1
            if participant_id not in part_lesion:
                part_lesion.append(participant_id)
         
        if ulcer_num > 0 or lesion_num > 0:
            concern_image_num += 1
            if participant_id not in part_concern:
                part_concern.append(participant_id)

    sta_file.close()
    views = ['dorsal', 'plantar', 'medial', 'lateral', 'toetips', 'heel']
    anno_type = ['ulcer', 'lesion', 'healed scar']
    for i,view in enumerate(views):
        print(f"ulcer: \"{view}\" num: {ulcer_view[i]}, percentage: {round(ulcer_view[i]/(np.sum(ulcer_view)-ulcer_view[-1])*100, 2)}%")
        print(f"lesion: \"{view}\" num: {lesion_view[i]}, percentage: {round(lesion_view[i]/(np.sum(lesion_view)-lesion_view[-1])*100, 2)}%")

    print(f"ulcer participant num: {len(part_ulcer)}")
    print(f"lesion participant num: {len(part_lesion)}")
    print(f"concern participant num: {len(part_concern)}")
    
    print(f"ulcer image num: {ulcer_image_num}, percentage: {round(ulcer_image_num/3362*100, 2)}%")
    print(f"lesion image num: {lesion_image_num}, percentage: {round(lesion_image_num/3362*100, 2)}%")
    print(f"concern image num: {concern_image_num}, percentage: {round(concern_image_num/3362*100, 2)}%")

if __name__ == "__main__":
    main()
    