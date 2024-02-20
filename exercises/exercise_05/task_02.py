import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#			Exercise 05			Task 2a									#
#########################################################################
print("\nTask 2a")
start_time = time.time()

def mean_filter(image, kernel_size):
    image_array = np.array(image)
    rows, columns = np.shape(image_array)

    # assume odd kernelsize
    # and no padding to initial image
    pad = int(kernel_size - 1)/2

    new_size_rows = int(rows - 2*pad)
    new_size_columns = int(columns - 2*pad)

    filtered_image = np.empty([new_size_rows, new_size_columns])
    
    #print(rows, columns)
    #print(np.shape(filtered_image))

    for row in range(new_size_rows):
        for column in range(new_size_columns):

            #print(image_array[row:row+kernel_size, column:column+kernel_size])
            new_value = np.mean(image_array[row:row+kernel_size, column:column+kernel_size])

            filtered_image[row, column] = new_value
    
    return(filtered_image)

def geometric_filter(image, kernel_size):

    image_array = np.array(image)
    rows, columns = np.shape(image_array)

    # and no padding to initial image -> resulting image will be a bit smaller
    pad = int(kernel_size - 1)/2

    new_size_rows = int(rows - 2*pad)
    new_size_columns = int(columns - 2*pad)

    filtered_image = np.empty([new_size_rows, new_size_columns])
    
    #print(rows, columns)
    #print(np.shape(filtered_image))

    for row in range(new_size_rows):
        for column in range(new_size_columns):
            
            
            
            submask = image_array[row:row+kernel_size, column:column+kernel_size]
            # this is important, else the product gives soemthing wrong as output
            submask = submask.astype(float)
            # for better runtime optimize whith taking ln on both sides
            # and then use convolve2d insted of 2 for loops
            product = np.prod(submask)
            filter_area = kernel_size * kernel_size
            new_value = product**(1/filter_area)
            
            # because of prod here we get some large numbers -> scale down to 1 ?
            # or limit filter
            # or extend to max float?
            # gets crazy even with 3x3 filter

            filtered_image[row, column] = new_value
    
    return(filtered_image)

def harmonic_filter(image, kernel_size):
    
    image_array = np.array(image)
    rows, columns = np.shape(image_array)

    # and no padding to initial image -> resulting image will be a bit smaller
    pad = int(kernel_size - 1)/2

    new_size_rows = int(rows - 2*pad)
    new_size_columns = int(columns - 2*pad)

    filtered_image = np.empty([new_size_rows, new_size_columns])
    
    #print(rows, columns)
    #print(np.shape(filtered_image))

    for row in range(new_size_rows):
        for column in range(new_size_columns):
            
            submask = image_array[row:row+kernel_size, column:column+kernel_size]
            # this is important, else the product gives soemthing wrong as output
            submask = submask.astype(float)
            filter_area = kernel_size * kernel_size
            epsilon = 0.00001
            # for not get in trouble when dividing by zero
            new_value = filter_area/np.sum(1/(submask+epsilon))

            filtered_image[row, column] = new_value
    
    return(filtered_image)

# main starts here

print("Give Kernelsize. Size should be an odd number")
kernel_size = int(input())

image_names = ["data\DIP3E_Original_Images_CH05\Fig0507(b)(ckt-board-gauss-var-400).tif",
"data\DIP3E_Original_Images_CH05\Fig0508(a)(circuit-board-pepper-prob-pt1).tif"]

for image_name in image_names:
    image = Image.open(image_name)

    if kernel_size % 2 == 0:
        print("Kernelsize has to be an odd number!")
        exit()

    if kernel_size > np.shape(image)[0] or kernel_size > np.shape(image)[1]:
        print("Kernelsize must be smaller than initial imagesize!")
        print("The imagesize is {}".format(np.shape(image)))
        print("but the chosen Kernelsize is {}".format(kernel_size))
        exit()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    fig.suptitle("Spatial Filters")

    axs[0,0].set_title("Original")
    axs[0,0].imshow(image, cmap="gray", vmin = 0, vmax = 255)
    axs[0,1].set_title("Mean-Filter with Kernelsize {}".format(kernel_size))
    axs[0,1].imshow(mean_filter(image, kernel_size), cmap="gray", vmin = 0, vmax = 255)
    axs[1,0].set_title("Geometric-Filter with Kernelsize {}".format(kernel_size))
    axs[1,0].imshow(geometric_filter(image, kernel_size), cmap="gray", vmin = 0, vmax = 255)
    axs[1,1].set_title("Harmonic-Filter with Kernelsize {}".format(kernel_size))
    axs[1,1].imshow(harmonic_filter(image, kernel_size), cmap="gray", vmin = 0, vmax = 255)
    plt.tight_layout()
    plt.show()

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 05			Task 2b									#
#########################################################################
print("\nTask 2b")
start_time = time.time()

def alpha_trimm_filter(image, kernel_size, d):
    
    image_array = np.array(image)
    rows, columns = np.shape(image_array)

    # and no padding to initial image -> resulting image will be a bit smaller
    pad = int(kernel_size - 1)/2

    new_size_rows = int(rows - 2*pad)
    new_size_columns = int(columns - 2*pad)

    filtered_image = np.empty([new_size_rows, new_size_columns])
    
    #print(rows, columns)
    #print(np.shape(filtered_image))

    for row in range(new_size_rows):
        for column in range(new_size_columns):
            
            submask = image_array[row:row+kernel_size, column:column+kernel_size]
            # this is important, else the product gives soemthing wrong as output
            submask = submask.astype(float)
            # flatten and sort submask
            submask = np.sort(submask.flatten())
            
            # remove d/2 lowest and highest intensity values
            submask = submask[int(d/2):-int(d/2)]

            filter_area = kernel_size * kernel_size
        
            new_value = 1/(filter_area - d) * np.sum(submask)

            filtered_image[row, column] = new_value
    
    return(filtered_image)

# main starts here

print("Give Kernelsize. Size should be an odd number")
kernel_size = int(input())

# d should be even and in range 0 and kernelsize*kernelsize
d = 4

image_names = ["data\DIP3E_Original_Images_CH05\Fig0507(b)(ckt-board-gauss-var-400).tif",
"data\DIP3E_Original_Images_CH05\Fig0508(a)(circuit-board-pepper-prob-pt1).tif"]

for image_name in image_names:
    image = Image.open(image_name)

    if not 0 <= d < kernel_size*kernel_size:
        print("d must be in range 0 to (kernelsize*kernelsize) - 1")
        exit()

    if kernel_size % 2 == 0:
        print("Kernelsize has to be an odd number!")
        exit()

    if kernel_size > np.shape(image)[0] or kernel_size > np.shape(image)[1]:
        print("Kernelsize must be smaller than initial imagesize!")
        print("The imagesize is {}".format(np.shape(image)))
        print("but the chosen Kernelsize is {}".format(kernel_size))
        exit()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 7))
    fig.suptitle("Spatial Filters")

    axs[0].set_title("Original")
    axs[0].imshow(image, cmap="gray", vmin = 0, vmax = 255)
    axs[1].set_title("Alpha-Trimm-Filter with Kernelsize {}".format(kernel_size))
    axs[1].imshow(alpha_trimm_filter(image, kernel_size, d), cmap="gray", vmin = 0, vmax = 255)
    
    plt.tight_layout()
    plt.show()

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))