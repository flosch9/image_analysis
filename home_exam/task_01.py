import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#########################################################################
#                                                                       #
#       Task 1               Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################
print("\nTask 1a")
start_time = time.time()

#########################################################################
#                               1                                       #
#########################################################################



image_name = "home_exam\pink-northern-lights.jpg"

image = np.asarray(Image.open(image_name))

print("Size of image:", image.shape)

# select two boxes just estimate the values based on the image size and looking at it
image_box_aurora = image[0:600,0:1200,:]
image_box_foreground = image[650:800,0:1000,:]

# get histograms of color image
# TODO check color range is it 255?
# maybe use this later
"""
image_hist_r = np.histogram(image[:,:,0], bins = 255)
image_hist_g = np.histogram(image[:,:,1], bins = 255)
image_hist_b = np.histogram(image[:,:,2], bins = 255)

image_box_aurora_hist_r = np.histogram(image_box_aurora[:,:,0], bins = 255)
image_box_aurora_hist_g = np.histogram(image_box_aurora[:,:,1], bins = 255)
image_box_aurora_hist_b = np.histogram(image_box_aurora[:,:,2], bins = 255)

image_box_foreground_hist_r = np.histogram(image_box_foreground[:,:,0], bins = 255)
image_box_foreground_hist_g = np.histogram(image_box_foreground[:,:,1], bins = 255)
image_box_foreground_hist_b = np.histogram(image_box_foreground[:,:,2], bins = 255)
"""

# define function for plotting
def plot_image_with_hists(image, titel,):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    fig.suptitle(titel)

    axs[0,0].set_title("Image")
    axs[0,0].imshow(image)
    axs[0,1].set_title("Red channel")
    axs[0,1].hist(image[:,:,0].flatten(), bins = 255)
    axs[1,0].set_title("Green channel")
    axs[1,0].hist(image[:,:,1].flatten(), bins = 255)
    axs[1,1].set_title("Blue channel")
    axs[1,1].hist(image[:,:,2].flatten(), bins = 255)
    fig.tight_layout()
    plt.show()
    
    return()

plot_image_with_hists(image,"Original image")
plot_image_with_hists(image_box_aurora,"Selected region of aurora")
plot_image_with_hists(image_box_foreground,"Selected region of foreground")


#########################################################################
#                               2                                       #
#########################################################################

def hist_equlize():
    return()

#########################################################################
#                               3                                       #
#########################################################################

def rgb_to_hsi(rgb_image):
    hsi_image = 1
    return(hsi_image)

#########################################################################
#                               4                                       #
#########################################################################

def mahalanobis_distance(z, a, c):
    distance = (z-a).T*c*(z-a)
    return(distance)

def image_select_mahalanobis_distance(whole_image, foreground_image, threshold):
    a_mean = np.mean(foreground_image)
    covariance = 1

    distance = mahalanobis_distance(whole_image, a_mean, covariance)
    
    selected_image = np.where(distance > threshold, whole_image)

    return(selected_image, distance)



#########################################################################
#                               5                                       #
#########################################################################



end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))