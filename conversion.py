'''
Python Script written by Anthony Nguyen
Usage:

WARNING: Be sure to change source_path and destination_path below - To find path, use os.getcwd() from os
WARNING1: Be sure that the source_path contains all images and destination_path be an empty folder  
The python script takes all the images in source_path and converts them to num_rows x num_cols and saves them in destination_path

Parameter: The resulting size of the image num_rows x num_cols

How to Run: 
    
WARNING: 1 conversion type at a time
    
Specify below 1) num_rows, 2) num_cols, 3) source_path - folder of all the images (no other file types!) 4) destination_path - an empty folder where the modified images are to be stored

Input names:  
    
num_rows: number of rows of resized image
num_cols: number of columns of resized image

resize: If set to True, images are resized to num_rows by num_cols. If set False, pass any numerical values of num_rows and num_cols.

horizontal_reflection: Set True for images to be horizonatally flipped
rotate_image: Set True for rotation to be enabled
degree: If rotate_image is set to be True, specify how many degrees of rotation for the images.

gauss_noise: If set to True, we add random noise to images. (Careful: This can make the conversion slower)
mean: If gauss_noise is set to True, random noise drawn from gaussian noise with mu = mean
std_dev: If gauss_noise is set to True, random noise drawn from gaussian noise with sigma = std_dev

shear_horizontal: If set to True, horizontally shear images.
horizontal_factor: If shear_horizontal is set to True, horizontal shear images by factor horizontal_factor.

shear_vertical: If set to True, vertically shear images.
vertical_factor: If shear_vertical is set to True, vertically shear images by factor vertical_factor.

zero_padding: If set to True, pad the images by zeros and resulting image is padding_dim by padding_dim
padding_dim: If zero_padding is set to True, the padding images are dimension padding_dim by padding_dim

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
from PIL import ImageOps

# Function that resize all images in source_path to dimension num_rows by num_cols and saves them in destination_path 
def convert_image(source_path,destination_path,num_rows,num_cols,resize,horizontal_reflection,rotate_image,degree,gauss_noise,mean,std_dev,shear_horizontal,horizontal_factor,shear_vertical,vertical_factor,zero_padding,padding_dim):

    path = [source_path,destination_path]
    
    # Get all the file names in the folder
    list_files = os.listdir(path[0])
    # Number of files in source path
    num_files = len(list_files)   

    # Counter for files processed
    files_processed = 1    
    
    # Iterate over all the images, perform averaging over the RGB channels, rescale to N x N, and save image in destination    
    for file in list_files:
        print(str(files_processed) + ' files processed out of ' + str(num_files) + ' files.')
        files_processed += 1
        # Take an image in name folder and open it
        # Then resize the image to N x N
        image = Image.open(path[0] + '\\' + file)
        # Try averaging over colored image
        try:
            # Averaging images over RGB channels
            image_array = np.array(image)
            image_average = np.zeros((np.shape(image_array)[0],np.shape(image_array)[1]))
            for i in range(3):
                image_average += image_array[:,:,i]
            image_average /= 3
                
            # Preprocessing so every image is in the format 0-255 by np.uint8
            image_average = image_average.astype(np.uint8)
            image = Image.fromarray(image_average)
        # Gray scale images    
        except IndexError: 
            pass
        # Reflection
        if horizontal_reflection == True:
            image = ImageOps.mirror(image)
        # Rotation
        if rotate_image == True:
            image = image.rotate(degree)
        # Resizing 
        if resize == True:
            image = image.resize((num_rows,num_cols))
        # Gaussian Noise
        if gauss_noise == True:
            img = np.array(image)  
            # add Gaussian Noise
            noisy_img = img + np.random.normal(mean,std_dev,img.shape)
            # remove out of range pixel values
            noisy_img_clipped = np.clip(noisy_img,0,255)
            # reformat to get image to render
            noisy_img_clipped_format = noisy_img_clipped.astype(np.uint8)
            image = Image.fromarray(noisy_img_clipped_format)
        # Shearing Images
        if shear_horizontal == True:
            image = image.transform(image.size,Image.AFFINE,(1,horizontal_factor,0,0,1,0))
        if shear_vertical == True:
            image = image.transform(image.size,Image.AFFINE,(1,0,0,vertical_factor,1,0))
        if zero_padding == True:
            if np.shape(image)[0] < padding_dim and np.shape(image)[1] < padding_dim:
                zero_matrix = np.zeros((padding_dim,padding_dim))
                image_matrix = np.array(image)
                zero_matrix[((padding_dim-image_matrix.shape[0])//2):image_matrix.shape[0]+(((padding_dim-image_matrix.shape[0])//2)),(((padding_dim-image_matrix.shape[1])//2)):image_matrix.shape[1]+(((padding_dim-image_matrix.shape[1])//2))] = image_matrix
                zero_matrix_format = zero_matrix.astype(np.uint8)
                image = Image.fromarray(zero_matrix_format)
            else:
                print('no padding for image' + file)
        # Save image in destination folder in format .jpg
        image.save(path[1] + '\\' + file,'JPEG')
    if resize == True:
        print('\nYour images are resized at ' + str(num_rows) + ' by ' + str(num_cols) + ' and are in the path ' + destination_path)
    
# Specify paths
    
train_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train'
train_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train_modified'
test_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test'
test_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test_modified'

# Run the image conversion
convert_image(train_path,train_destination_path,1000,500,False,False,False,10,False,50,1,False,0.2,False,0.2,False,2500)
