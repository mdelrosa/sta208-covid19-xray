'''
Python Script written by Anthony Nguyen
Usage:

WARNING: Be sure to change source_path and destination_path below - To find path, use os.getcwd() from os
WARNING1: Be sure that the source_path contains all images and destination_path be an empty folder  
The python script takes all the images in source_path and converts them to num_rows x num_cols and saves them in destination_path

Parameter: The resulting size of the image num_rows x num_cols

How to Run: 
    
Specify below 1) num_rows, 2) num_cols, 3) source_path - folder of all the images (no other file types!) 4) destination_path - an empty folder where the modified images are to be stored

Input names:  
    
num_rows: number of rows of resized image
num_cols: number of columns of resized image

Pick exactly one (1) pair below: source_path in [train_path,test_path] and destination_path in [train_destination_path,test_destination_path]

train_path = path containing all training images
train_destination_path = path where resized training images are to be saved

test_path = path containing all testing images
test_destination_path = path where resized test images are to be saved

'''

# Import libraries
import numpy as np
import os
from PIL import Image

# Function that resize all images in source_path to dimension num_rows by num_cols and saves them in destination_path 
def convert_image(source_path,destination_path,num_rows,num_cols):

    path = [source_path,destination_path]
    
    # Get all the file names in the folder
    list_files = os.listdir(path[0])
    # Number of files in source path
    num_files = len(list_files)
    
    # Files that are problematic and need to be manually processed
    #list_files_exception = ['acute-respiratory-distress-syndrome-ards.jpg','acute-respiratory-distress-syndrome-ards-1.jpg','SARS-10.1148rg.242035193-g04mr34g07b-Fig7b-day12.jpeg','SARS-10.1148rg.242035193-g04mr34g09a-Fig9a-day17.jpeg','SARS-10.1148rg.242035193-g04mr34g09b-Fig9b-day19.jpeg','SARS-10.1148rg.242035193-g04mr34g09c-Fig9c-day27.jpeg']

    # Counter for files processed
    files_processed = 1    
    
    # Iterate over all the images, perform averaging over the RGB channels, rescale to N x N, and save image in destination    
    for file in list_files:
        print(str(files_processed) + ' files processed out of ' + str(num_files) + ' files.')
        files_processed += 1
        # Take an image in name folder and open it
        # Then resize the image to N x N
        image = Image.open(path[0] + '\\' + file)
        image2 = image.resize((num_rows,num_cols))
            
        # Try averaging over colored image
        try:
            # Averaging images over RGB channels
            image2_array = np.array(image2)
            image_average = np.zeros((np.shape(image2_array)[0],np.shape(image2_array)[1]))
            for i in range(3):
                image_average += image2_array[:,:,i]
            image_average /= 3
                
            # Preprocessing so every image is in the format 0-255 by np.uint8
            # Save image in destination folder in format .jpg
            image_average = image_average.astype(np.uint8)
            image3 = Image.fromarray(image_average)
            image3.save(path[1] + '\\' + file,'JPEG') 
        # Gray scale images    
        except IndexError: 
            image2.save(path[1] + '\\' + file,'JPEG')
    print('\nYour images are resized at ' + str(num_rows) + ' by ' + str(num_cols) + ' and are in the path ' + destination_path)
    
# Specify paths
    
train_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train'
train_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train_modified'
test_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test'
test_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test_modified'

# Run the image conversion
convert_image(test_path,test_destination_path,1000,500)