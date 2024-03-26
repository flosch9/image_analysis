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

# define function for selecting a box from the given image 
def select_image_box(image : np.ndarray, anchor : tuple, 
                     size : tuple , savename : str = None,  
                     showplot : bool = True):

    if anchor[0]+size[0]>image.shape[0] or anchor[1]+size[1]>image.shape[1]:
        print("Selected anchor or box size is not within the image.\nImage has size {}".format(image.shape))
        pass
    
    selected_image_box = image[anchor[0]:anchor[0]+size[0], anchor[1]:anchor[1]+size[1]]

    box_anchor = (anchor[1], anchor[0])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))
    #fig.suptitle("Original Image with selected box")

    axs[0].set_title("Whole Image with selected area")
    axs[0].imshow(image)
    axs[0].add_patch(Rectangle(box_anchor,size[1],size[0],linewidth=1,edgecolor='r',facecolor='none'))
    axs[1].set_title("Selected area")
    axs[1].imshow(selected_image_box)#.resize(image.shape))
    fig.tight_layout()
    
    if savename != None:
        plt.savefig(savename)
    
    if showplot:
        plt.show()
    
    plt.close()

    return(selected_image_box)

# select two boxes just estimate the values for the anchor and the size
# based on the image size and looking at it

aurora_box_anchor = (260,60)
aurora_box_size = (300,1000)

foreground_anchor = (630,100)
foreground_size = (150,1000)

# first tuple is anchor (y,x) on the upper left corner of the selected box
# second tuple is height and width (delta_y, delta_x) of the selected box
image_box_aurora = select_image_box(image, aurora_box_anchor, aurora_box_size, savename= "task01_a_box_aurora.png")
image_box_foreground = select_image_box(image, foreground_anchor, foreground_size, savename= "task01_a_box_foreground.png")


# define function for plotting images with histograms
def plot_image_with_hists(image : np.ndarray, maintitel : str, 
                          subtitles  :list = 
                          ["Image", "Red channel", "Green channel", "Blue channel"], savename : str = None,
                          showplot : bool = True):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    #fig.suptitle(maintitel)

    axs[0,0].set_title(subtitles[0])
    axs[0,0].imshow(image)
    axs[0,1].set_title(subtitles[1])
    axs[0,1].hist(image[:,:,0].flatten(), bins = 256)
    axs[1,0].set_title(subtitles[2])
    axs[1,0].hist(image[:,:,1].flatten(), bins = 256)
    axs[1,1].set_title(subtitles[3])
    axs[1,1].hist(image[:,:,2].flatten(), bins = 256)
    fig.tight_layout()

    if savename != None:
        plt.savefig(savename)
    
    if showplot:
        plt.show()
    
    plt.close()
    
    return()

plot_image_with_hists(image,"Original image", savename="task01_a_whole_image.png")
plot_image_with_hists(image_box_aurora,"Selected region of aurora", savename="task01_a_aurora.png")
plot_image_with_hists(image_box_foreground,"Selected region of foreground",savename="task01_a_foreground.png")


#########################################################################
#                               2                                       #
#########################################################################


def hist_equalize(image : np.ndarray):
    
    new_image = np.zeros(image.shape)

    for channel in range(3):

        equalized_channel = cv.equalizeHist(image[:,:,channel])
       
        new_image[:,:,channel] = equalized_channel#/255
        
    # return astype int is important, else warning for clipping data
    # and later for segmentation errors
    return(new_image.astype(int))

# equalize histogram
equalized_image = hist_equalize(image)

# select same areas as before from equalized image
equalized_image_box_aurora = select_image_box(equalized_image, aurora_box_anchor, aurora_box_size, showplot=False)
equalized_image_box_foreground = select_image_box(equalized_image, foreground_anchor, foreground_size, showplot=False)


plot_image_with_hists(equalized_image, "Whole image with histogram equalization", savename="task01_b_whole_image.png")
plot_image_with_hists(equalized_image_box_aurora, "Aurora region with histogram equalization", savename="task01_b_box_aurora.png")
plot_image_with_hists(equalized_image_box_foreground, "Foreground image with histogram equalization", savename="task01_b_box_foreground.png")



#########################################################################
#                               3                                       #
#########################################################################

# define function for transformatin rgb to hsi
# very close to the exercise function
def rgb_to_hsi(rgb_image : np.ndarray):
    # epsilon for not getting in to trouble with small numbers while dividing
    epsilon = 1e-5
    
    R,G,B = rgb_image[:,:,0],rgb_image[:,:,1],rgb_image[:,:,2]
    
    arg = 0.5*((R-G)+(R-B)) / (np.sqrt((R-G)**2 + (R-B)*(G-B))+ epsilon)

    # important for having reasonabel values for arccos [-1,1]
    arccos_argument = np.where(arg < -1, -1, arg)
    arccos_argument = np.where(arg > 1, 1, arg)

    theta = np.arccos(arccos_argument)
    theta *= 180/np.pi 

    H = np.copy(theta)
    H[B>G] = 360-theta[B>G]
    
    S = 1-3*np.min(rgb_image, axis=2)/(np.sum(rgb_image, axis=2) + epsilon)
    
    I = np.mean(rgb_image, axis=2)
    
    hsi_image = np.stack([H,S,I], axis=2)
    return(hsi_image)

"""
# not used
# define backtransformation hsi to rgb
def hsi_to_rgb(hsi_image):

    H,S,I = hsi_image[:,:,0], hsi_image[:,:,1], hsi_image[:,:,2]

    if H < 0:
        H += 2 * np.pi
    elif H >= 2 * np.pi:
        H -= 2 * np.pi

    if H < 2 * np.pi / 3:
        B = I * (1 - S)
        R = I * (1 + (S * np.cos(H)) / (np.cos(np.pi / 3 - H)))
        G = 3 * I - (R + B)
    elif H < 4 * np.pi / 3:
        H -= 2 * np.pi / 3
        R = I * (1 - S)
        G = I * (1 + (S * np.cos(H)) / (np.cos(np.pi / 3 - H)))
        B = 3 * I - (R + G)
    else:
        H -= 4 * np.pi / 3
        G = I * (1 - S)
        B = I * (1 + (S * np.cos(H)) / (np.cos(np.pi / 3 - H)))
        R = 3 * I - (G + B)

    rgb_image = np.stack([R,G,B], axis=2)

    return(rgb_image)
"""


hsi_image = rgb_to_hsi(image)
hsi_image_box_aurora = rgb_to_hsi(image_box_aurora)
hsi_image_box_foreground = rgb_to_hsi(image_box_foreground)

plot_image_with_hists(hsi_image,"To hsi converted image", ["HSI-Image", "H Channel", "S Channel", "I channel"], savename="task01_c_whole_image.png")
plot_image_with_hists(hsi_image_box_aurora,"To hsi converted aurora image", ["HSI-Image", "H Channel", "S Channel", "I channel"],savename="task01_c_box_aurora.png")
plot_image_with_hists(hsi_image_box_foreground,"To hsi converted foreground image", ["HSI-Image", "H Channel", "S Channel", "I channel"], savename="task01_c_box_foreground.png")


#########################################################################
#                               5                                       #
#########################################################################

def mahalanobis_distance(z, a, c):
    # pixel z
    # mean a
    # inverse covariance c
    distance = np.sqrt((z-a).T*c*(z-a))
    return(distance)


def image_select_mahalanobis_distance(whole_image, foreground_image, threshold_percent):
    # Compute the inverse of the covariance matrix
    # modification of shape .1, 1 and cov as scalar is better
    covariance = np.cov(foreground_image.reshape(-1, 3), rowvar=False)
    mean = np.mean(foreground_image, axis=(0, 1))

    inv_covariance = np.linalg.inv(covariance)
 
    # Compute Mahalanobis distance for each pixel
    distance = np.zeros_like(whole_image, dtype=np.float32)
    for i in range(whole_image.shape[0]):
        for j in range(whole_image.shape[1]):
            pixel = whole_image[i, j]
            difference = pixel - mean
            # definition of distance sqrt (with/without it?)
            dist = np.sqrt((np.dot(np.dot(difference.T, inv_covariance), difference)))
            distance[i, j] = dist


    print("\nRegion of distance, max/min")
    print(np.amax(distance))
    print(np.amin(distance))

    threshold = threshold_percent * np.amax(distance)

    segmented_foreground = np.where(distance >= threshold, whole_image, 255)
    segmented_background = np.where(distance < threshold, whole_image, 255)
  
    return(segmented_foreground, segmented_background, distance)


def plot_segmentation(segemented_foreground, segemented_background, distance, maintitle = "Image segmentation", title1 = "Segemented foreground", title2 = "Segemented background", title3 = "Distance as 2d image", savename : str = None, showplot : bool = True):

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 7))
    fig.suptitle(maintitle)

    axs[0].set_title(title1)
    axs[0].imshow(segemented_foreground)
    axs[1].set_title(title2)
    axs[1].imshow(segemented_background)
    axs[2].set_title(title3)
    axs[2].imshow(distance, cmap = "gray")
   
    fig.tight_layout()
    
    if savename != None:
        plt.savefig(savename)
    
    if showplot:
        plt.show()
    
    plt.close()
    
    return()

#segmented_image, distance = image_select_mahalanobis_distance(image, image_box_foreground, threshold=2)

#equalized_segmented_image, equalized_distance = image_select_mahalanobis_distance(equalized_image, equalized_image_box_foreground, threshold=2)

#hsi_segmented_image, hsi_distance = image_select_mahalanobis_distance(hsi_image, hsi_image_box_foreground, threshold=2)

# threshold in % of max distance
threshold_image = 0.2
threshold_equalized_image = 0.3
threshold_hsi_image = 0.2



plot_segmentation(*image_select_mahalanobis_distance(image, image_box_foreground, threshold_image), savename="task01_e_image_segementation.png")
plot_segmentation(*image_select_mahalanobis_distance(equalized_image, equalized_image_box_foreground, threshold_equalized_image), savename="task01_e_equalized_segementation.png")
plot_segmentation(*image_select_mahalanobis_distance(hsi_image, hsi_image_box_foreground, threshold_hsi_image), savename="task01_e_hsi_segementation.png")



#########################################################################
#                               6                                       #
#########################################################################


def image_select_mahalanobis_distance_improved(whole_image, foreground_image, threshold_percent):
    # Compute the inverse of the covariance matrix
    # modification of shape .1, 1 and cov as scalar is better
    covariance = np.cov(foreground_image.reshape(-1, 3), rowvar=False)
    mean = np.mean(foreground_image, axis=(0, 1))
  
    inv_covariance = np.linalg.inv(covariance)

    # Compute Mahalanobis distance for each pixel
    distance = np.zeros_like(whole_image, dtype=np.float32)
    for i in range(whole_image.shape[0]):
        for j in range(whole_image.shape[1]):
            pixel = whole_image[i, j]
            difference = pixel - mean
            # definition of distance sqrt (with/without it?)
            dist = np.sqrt((np.dot(np.dot(difference.T, inv_covariance), difference)))
            distance[i, j] = dist

    print("\nRegion of distance, max/min")
    print(np.amax(distance))
    print(np.amin(distance))

    threshold = threshold_percent * np.amax(distance)

    #distance = cv.GaussianBlur(distance, (31,31),0)
    distance = cv.blur(distance, (41,41))

    segmented_foreground = np.where(distance >= threshold, whole_image, 255)
    segmented_background = np.where(distance < threshold, whole_image, 255)

    return(segmented_foreground, segmented_background, distance)



plot_segmentation(*image_select_mahalanobis_distance_improved(image, image_box_foreground, threshold_image), savename="task01_f_image_segementation_improved.png")
plot_segmentation(*image_select_mahalanobis_distance_improved(equalized_image, equalized_image_box_foreground, threshold_equalized_image), savename="task01_f_equalized_segementation_improved.png")
plot_segmentation(*image_select_mahalanobis_distance_improved(hsi_image, hsi_image_box_foreground, threshold_hsi_image), savename="task01_f_hsi_segementation_improved.png")

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))