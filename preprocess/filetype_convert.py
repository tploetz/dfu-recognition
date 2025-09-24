import os

from PIL import Image

def jpeg2jpg(input_root):
    count = 0
# Loop through all files in the directory
    for filename in os.listdir(input_root):
        # print(filename)
        if filename.endswith('.jpeg'):
            count += 1
            # Construct the full file path
            file_path = os.path.join(input_root, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # print(f'Processing {filename}...')
                # Define the new filename (change extension to .jpg)
                new_filename = filename[:-5] + '.jpg'
                new_file_path = os.path.join(input_root, new_filename)
                
                # Save the image with the new extension
                img.save(new_file_path)

                # Optionally, delete the original .jpeg file
                os.remove(file_path)
            # print(f'Converted {filename} to {new_filename}')
    print(f'{input_root}, jpeg2jpg: {count}')

def jpg2png(input_root):
    count = 0
# Loop through all files in the directory
    for filename in os.listdir(input_root):
        # print(filename)
        if filename.endswith('.jpg'):
            count += 1
            # Construct the full file path
            file_path = os.path.join(input_root, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # print(f'Processing {filename}...')
                # Define the new filename (change extension to .jpg)
                new_filename = filename[:-4] + '.png'
                new_file_path = os.path.join(input_root, new_filename)
                
                # Save the image with the new extension
                img.save(new_file_path)

                # Optionally, delete the original .jpeg file
                os.remove(file_path)
    print(f'{input_root}, jpg2png: {count}')
def png2jpg(input_root):
    count = 0
    for filename in os.listdir(input_root):
        # print(filename)
        if filename.endswith('.png'):
            count += 1
            # Construct the full file path
            file_path = os.path.join(input_root, filename)
            
            # Open the image
            with Image.open(file_path) as img:
                # print(f'Processing {filename}...')
                # Define the new filename (change extension to .jpg)
                new_filename = filename[:-4] + '.jpg'
                new_file_path = os.path.join(input_root, new_filename)
                
                # Save the image with the new extension
                img.save(new_file_path)

                # Optionally, delete the original .jpeg file
                os.remove(file_path)
    print(f'{input_root}, png2jpg: {count}')