'''
Python Script written by Anthony Nguyen

Usage: Make sure to comment function not used in {convert_image,list_unique_x_ray}

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
How to Run function convert_image

WARNING: Be sure to change source_path and destination_path below - To find path, use os.getcwd() from os
WARNING1: Be sure that the source_path contains all images and destination_path be an empty folder  
The python script takes all the images in source_path and converts them to num_rows x num_cols and saves them in destination_path

Parameter: The resulting size of the image num_rows x num_cols
    
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

aspect: If set to True, rescale image while preserving aspect ratio. Resulting image with largest dimension is 'dimension'
dimension: If aspect is set to True, rescale image while preserving aspect ratio. Resulting image with largest dimension is 'dimension' while the smaller dimension may be strictly smaller than 'dimension'

label_dict: This is a dictionary of labels used by our Jupyter notebooks.
print_every=100: Printing related task in preprocessing.

Pick exactly one (1) pair below: source_path in [train_path,test_path] and destination_path in [train_destination_path,test_destination_path]

train_path = path containing all training images
train_destination_path = path where resized training images are to be saved

test_path = path containing all testing images
test_destination_path = path where resized test images are to be saved

Output: Modify images from source_path and saves them in destination_path

---------------------------------------------------------------------------------------------------------------------------------------------------------------

How to Run function list_unique_x_ray

NOTE: Colored images are automatically averaged across RGB channel and converted to gray scale.

Input names:  
    
    source_path: Path to take images
    destination_path: Path to save unique patients' x-ray
    rng: If True (set viral to False), randomly pick one x-ray from each patient and add to destination_path.
    viral: If True (set rng to False), pick the first viral observed from each patient and add to destination_path. If a viral x-ray does not exist, take a bacterial x-ray.
    normal_rng: If True, randomly pick exactly 1 x-ray from each patient
    
Output: Modify images from source_path and saves them in destination_path

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

'''

# Import libraries
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
import random

# Function that resize all images in source_path to dimension num_rows by num_cols and saves them in destination_path 
def convert_image(source_path,destination_path,num_rows,num_cols,resize,horizontal_reflection,rotate_image,degree,gauss_noise,mean,std_dev,shear_horizontal,horizontal_factor,shear_vertical,vertical_factor,zero_padding,padding_dim,aspect,dimension,label_dict,print_every=100):

    path = [source_path,destination_path]
    
    # Get all the file names in the folder
    list_files = os.listdir(path[0])
    # Number of files in source path
    num_files = len(list_files)   

    # Counter for files processed
    files_processed = 1  
    # List of bad x-rays
    bad_labels = []
    
    # Iterate over all the images, perform averaging over the RGB channels, rescale to N x N, and save image in destination    
    for file in list_files:
        # Print files processed every 100 x-rays
        if files_processed % print_every == 0:
            print(str(files_processed) + ' files processed out of ' + str(num_files) + ' files.')
        files_processed += 1
        # Take an image in name folder and open it
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
        # Aspect ratio rescaling
        if aspect == True:
            List = image.size
            index_max = List.index(max(List))
            index_min = List.index(min(List))
            aspect_ratio = List[index_max] / List[index_min]
            if index_max == 1:
                image = image.resize((int(dimension/ aspect_ratio),dimension))
            else:
                image = image.resize((dimension, int(dimension / aspect_ratio)))
        # Reflection
        if horizontal_reflection == True:
            image = ImageOps.mirror(image)
        # Rotation
        if rotate_image == True:
            image = image.rotate(degree)
        # Resizing
        # Then resize the image to num_rows x num_cols
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
        try:
            label_dict[file][0] # first element is label, if x-ray not in metadata.csv, go to except
            # Save image in destination folder in format .jpg
            image.save(path[1] + '\\' + file,'JPEG')
        except: # label not found -- skip and increment
            bad_labels.append(file)
    if resize == True:
        print('\nYour images are resized at ' + str(num_rows) + ' by ' + str(num_cols) + ' and are in the path ' + destination_path + '\n')
    print(f'--- There were {len(bad_labels)} images without labels ---')
    
def list_unique_x_ray(source_path,destination_path,rng,viral,normal_rng):

    list_files = os.listdir(source_path)
    
    # Process Unique x-rays from infected patients
    
    # get files with string 'person' in the file 
    list_files_person = []
    for file in list_files:
        if ('person' in file) or ('IM' in file) or ('NORMAL' in file):
            list_files_person.append(file)
            
    
    person_string = 'person'
    index = len('person')    
    
    # Append 1 x-ray per patient in list_unique_person_files
    list_unique_person_files = []
    person_x_rays = []
    for file in list_files_person:
        
        # Get Patient ID 
        stack = person_string
        j = index
        while file[j] != '_':
            stack += file[j]
            j += 1
        # Append x-ray corresponding to patient
        if not person_x_rays:
            person_x_rays.append(file)
        else:
            # check if current x-ray corresponds to the previous checked patient
            # If so, take an x-ray from the previously checked patient and push file in empty stack
            if stack in person_x_rays[-1]:
                person_x_rays.append(file)
            else:
                if rng:
                    # Randomly pick an xray from a person
                    random_index = random.randint(0,len(person_x_rays)-1)
                    list_unique_person_files.append(person_x_rays[random_index])
                if viral:
                    # Append the first virus x-ray coresponding to a patient. If not found, take a bacterial x-ray.
                    virus_count = 0
                    for item in person_x_rays:                       
                        if 'virus' in item:
                            virus_count += 1
                            list_unique_person_files.append(item)
                            break
                    if virus_count == 0:
                        list_unique_person_files.append(person_x_rays[0])                    
                # clear list to load x-rays for the next person
                person_x_rays = []
                # append file in person_x_rays
                person_x_rays.append(file)
                
    path = [source_path,destination_path]
    
    # Process Unique x-rays from normal patients
    
    # Stack holds x-rays from a given patient
    stack_normal_patient = []
    
    # Sort x-ray to an existing patient or clear stack and push x-ray in stack
    for file in list_files:
        
        # If stack is empty, push x-ray in stack
        if 'IM' in file and stack_normal_patient == []:
            stack_normal_patient.append(file)
            
            # If stack is empty andat the last file
            if file == list_files[-1]:
                list_unique_person_files.append(file)
            continue
        
        # If x-ray is valid
        if 'IM' in file:
            
            # If x-ray has string 'NORMAL', push it in the stack if it corresponds to the other x-rays in the stack or clear stack and push x-ray in stack
            if 'NORMAL2' in file:
                file_name = file.split('.')
                file_name.pop()
                file_name = '-'.join(file_name)
                file_name = file_name.split('-')
                file_name.pop(0)
                file_name.pop(0)
                if len(file_name) == 3:
                    file_name.pop()
                file_name = '-'.join(file_name)
                if file_name in stack_normal_patient[0]:
                    stack_normal_patient.append(file)
                else:
                    # If normal_rng == True, randomly pick an x-ray in stack_normal_patient and append to list_unique_person_files
                    if normal_rng:
                        list_unique_person_files.append(stack_normal_patient[random.randint(0,len(stack_normal_patient)-1)])
                    else:
                        list_unique_person_files.append(stack_normal_patient[0])
                    stack_normal_patient = []
                    stack_normal_patient.append(file)
            
            # Push in stack if x-ray corresponds to other x-rays in the stack or clear stack and push in stack
            else:
                file_name = file.split('.')
                file_name.pop()
                file_name = '-'.join(file_name)
                file_name = file_name.split('-')
                #file_name.pop(0)
                file_name.pop(0)
                if len(file_name) == 3:
                    file_name.pop()
                file_name = '-'.join(file_name)
                if file_name in stack_normal_patient[0]:
                    stack_normal_patient.append(file)
                else:
                    # If normal_rng == True, randomly pick an x-ray in stack_normal_patient and append to list_unique_person_files
                    if normal_rng:
                        list_unique_person_files.append(stack_normal_patient[random.randint(0,len(stack_normal_patient)-1)])
                    else:
                        list_unique_person_files.append(stack_normal_patient[0])
                    stack_normal_patient = []
                    stack_normal_patient.append(file)
                
                   
    
    # Number of files in source path
    num_files = len(list_unique_person_files)   
    
    
    # Counter for files processed
    files_processed = 1    
    
    # Iterate over all the images, perform averaging over the RGB channels, rescale to N x N, and save image in destination 
    print("--- Processing of unique {} files ---".format(num_files))
    for file in list_unique_person_files:
        if files_processed % print_every == 0:
            print(str(files_processed) + ' files processed out of ' + str(num_files) + ' files.')
        files_processed += 1
        # Take an image in name folder and open it
        image = Image.open(path[0] + os.path.sep + file)
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
        image.save(path[1] + os.path.sep + file,'JPEG')
     
def get_target_paths(base_path):
    train_path = f'{base_path}\\train'
    train_dest = f'{base_path}\\train_modified'
    test_path = f'{base_path}\\test'
    test_dest = f'{base_path}\\test_modified'
    return [train_path, train_dest, test_path, test_dest]        

        
def check_target_paths(target_paths, label_names):
    for path in target_paths:
        for label in label_names:
            try:
                os.stat(f'{path}\\{label}')
                print(f"-> Verified target path: {path}\\{label}")
            except:
                print(f"-> '{path}\\{label}' does not exist. Create path.")
                os.mkdir(f'{path}\\{label}')
        
        
if (__name__ == "__main__"):
    
    # Specify paths
        
    train_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train'
    train_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\train_modified'
    test_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test'
    test_destination_path = 'C:\\Users\\Tony Nguyen\\Desktop\\Coronahack_STAT208\\coronahack-chest-xraydataset\\Coronahack-Chest-XRay-Dataset\\Coronahack-Chest-XRay-Dataset\\test_modified'
    
    # Run the image conversion
    convert_image(train_path,train_destination_path,1000,500,False,False,False,10,False,50,1,False,0.2,False,0.2,True,600,True,595,label_dict,print_every=500)
    
    # Extract Unique x-rays from source_path to destination_path
    list_unique_x_ray(test_path,test_destination_path,True,False,True)

