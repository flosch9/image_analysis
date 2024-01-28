import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Task 2
print("\n Task 2")

image = Image.open("data\DIP3E_Original_Images_CH03\Fig0310(b)(washed_out_pollen_image).tif")

#image_float = np.array(image).astype(float)
# np.float gives a warning so this with just float as argument is better
# even better alternative is to use the following
image_float = np.array(image, dtype = "float64")

# Testing and comparisaon
# python float, which float in IEE754 is used: 64bit (guess so)
print("Uint8 (8 bit fixed) image: \n", np.array(image))
print("Double (float64) image: \n", image_float)

image_fixed = image_float.astype(np.uint8)

# im2double function
def scale_to_one(image_array):
    # check for type:
    if image_array.dtype is not np.dtype("float64"):
        print("WARNING: arry should be of numpy dtype 'float64'.")
        print("Given array is of type: ", image_array.dtype)
    
    # if we use for example int as input it should also work 
    # because python converts automaticaly to float so just go on

    scaled_image_array = image_array/255
    return(scaled_image_array)


def scale_to_integer(image_array):
    int_image_array = image_array   
    return(int_image_array)

print("Scaled image:\n", scale_to_one(image_float))

# Testing
print(np.max(scale_to_one(image_float)))
print(np.max(image_fixed))