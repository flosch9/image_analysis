import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def select_image_box(image, anchor : tuple, size : tuple , savename = None):

    if anchor[0]+size[0]>image.shape[0] or anchor[1]+size[1]>image.shape[1]:
        print("Selected anchor or box size is not within the image.\nImage has size {}".format(image.shape))
        pass
    
    selected_image_box = image[anchor[0]:anchor[0]+size[0], anchor[1]:anchor[1]+size[1]]


    box_anchor = (anchor[1], anchor[0])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))
    fig.suptitle("Original Image with selected box")

    axs[0].set_title("Image with selected area")
    axs[0].imshow(image)
    axs[0].add_patch(Rectangle(box_anchor,size[1],size[0],linewidth=1,edgecolor='r',facecolor='none'))
    axs[1].set_title("Selected area")
    axs[1].imshow(selected_image_box)#.resize(image.shape))
    fig.tight_layout()
    plt.show()



    return(selected_image_box)

# select two boxes just estimate the values based on the image size and looking at it


# first tuple is anchor (y,x) on the upper left corner of the selected box
# second tuple is height and width (delta_y, delta_x) of the selected box
image_box_aurora = select_image_box(image, (260,60), (300,1000))
image_box_foreground = select_image_box(image, (630,100), (150,1000))

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

#def hist_equalize():
#    return()

# pre implemented functions (e.g. cv2) are also fine
def hist_equalize(image):
    new_image = np.zeros(image.shape)
    for channel in range(3):
        hist, _ = np.histogram(image[:,:,channel].flatten(), 256, [0, 255])
        cdf = hist.cumsum()
        cdf_norm = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        channel_new = cdf_norm[image[:,:,channel].flatten()]
        new_image[:,:,channel] = np.reshape(channel_new, image[:,:,channel].shape)
    return(new_image)


plot_image_with_hists(hist_equalize(image), "Whole image with histogram equalization.")
plot_image_with_hists(hist_equalize(image_box_aurora), "Aurora region with histogram equalization.")
plot_image_with_hists(hist_equalize(image_box_foreground), "Foreground image with histogram equalization.")



#########################################################################
#                               3                                       #
#########################################################################

# pre implemented functions (e.g. cv2) are also fine
def rgb_to_hsi(rgb_image):
    '''Function to convert an RGB image to an HSI image, from the exercises'''
    ## Scale the image down to the range [0,1]
    R,G,B = rgb_image[:,:,0],rgb_image[:,:,1],rgb_image[:,:,2]
    ## From equation 6-17
    theta = np.arccos( 0.5*((R-G)+(R-B)) / (np.sqrt((R-G)**2 + (R-B)*(G-B)) + 1e-8)
    )
    theta *= 180/np.pi # Convert the angle into degrees
    ## From equation 6-16
    H = np.copy(theta)
    H[B>G] = 360-theta[B>G]
    ## From equation 6-18
    S = 1-3*np.min(rgb_image, axis=2)/np.sum(rgb_image, axis=2)
    ## From equation 6-19
    I = np.mean(rgb_image, axis=2)
    ## return HSI image
    hsi_image = np.stack([H,S,I], axis=2)
    return(hsi_image)

hsi_image = rgb_to_hsi(image)
hsi_image_box_aurora = rgb_to_hsi(image_box_aurora)
hsi_image_box_foreground = rgb_to_hsi(image_box_foreground)

plot_image_with_hists(hsi_image,"To hsi converted image")
plot_image_with_hists(hsi_image_box_aurora,"To hsi converted aurora image")
plot_image_with_hists(hsi_image_box_foreground,"To hsi converted foreground image")



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

#########################################################################
#                               6                                       #
#########################################################################

#########################################################################
#                               7                                       #
#########################################################################



end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))