import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image


picture = Image.open(r"data\DIP3E_Original_Images_CH02\Fig0222(b)(cameraman).tif")

picture_array = np.array(picture)

# Task 6a
print("\n Task 6a")

def mean_pixel( start_x, start_y, pixels_x, pixels_y, image):
    mean_pixel = np.mean([image[x][y] for x,y in zip(range(int(start_x), int(start_x + pixels_x)), range(int(start_y), int(start_y + pixels_y)))])
    return(mean_pixel)

def simple_scale_by_factor(scaling_factor, image):
    rows, columns = np.shape(image)
    
    scaled_image = [[image[int(row/scaling_factor)][int(column/scaling_factor)] for   column in range(int(columns*scaling_factor))] for row in range(int(rows*scaling_factor))]
    # works for upscaling

    # problem when downscaling then also unsharpen -> build mean of pixels, is better
    
    return(scaled_image)

scaled_image = simple_scale_by_factor(2, picture_array)
plt.imshow(picture, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(scaled_image, cmap='gray', vmin=0, vmax=255)
plt.show()
# here used right with colormap grey

# Task 6b
print("\n Task 6b")

def rotate_by_rad(rad, image):
    rows, columns = np.shape(image)

    mid_x = int(columns/2)
    mid_y = int(rows/2)
    
    rotated_image = [[image[int((row-mid_y)*np.cos(rad) + (column-mid_x)*np.sin(rad)+mid_y)][int((column-mid_x)*np.sin(rad) + (row-mid_x)*np.cos(rad)+mid_x)] for column in range(int(columns))] for row in range(int(rows))]
    
    return(rotated_image)
#test

rotated_image = rotate_by_rad(0.5, picture_array)

plt.imshow(picture, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(rotated_image, cmap='gray', vmin=0, vmax=255)
plt.show()

# Task 6c
print("\n Task 6c")

def translation_by_pixels(trans_x, trans_y, image):
    rows, columns = np.shape(image)
    translated_image = [[image[row-trans_x][column-trans_y] for column in range(columns) ] for row in range(rows)]
    # get rid of the "overlap", change it to black? 
    return(translated_image)

translated_image = translation_by_pixels(100, 100, picture_array)

plt.imshow(picture, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(translated_image, cmap='gray', vmin=0, vmax=255)
plt.show()
# get rid of the "overlap", change it to black? 

# Task 6d
print("\n Task 6d")