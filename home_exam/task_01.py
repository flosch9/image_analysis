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

# define function for plotting
def plot_image_with_hists(image, maintitel, title1 = "Image", title2 = "Red channel", title3 = "Green channel", title4 = "Blue channel"):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    fig.suptitle(maintitel)

    axs[0,0].set_title(title1)
    axs[0,0].imshow(image)
    axs[0,1].set_title(title2)
    axs[0,1].hist(image[:,:,0].flatten(), bins = 255)
    axs[1,0].set_title(title3)
    axs[1,0].hist(image[:,:,1].flatten(), bins = 255)
    axs[1,1].set_title(title4)
    axs[1,1].hist(image[:,:,2].flatten(), bins = 255)
    fig.tight_layout()
    plt.show()
    
    return()

#plot_image_with_hists(image,"Original image")
#plot_image_with_hists(image_box_aurora,"Selected region of aurora")
#plot_image_with_hists(image_box_foreground,"Selected region of foreground")


#########################################################################
#                               2                                       #
#########################################################################

#def hist_equalize():
#    return()

# pre implemented functions (e.g. cv2) are also fine

# image should be in hsi space !!!

def rgb_to_hsi(rgb_image):

    '''Function to convert an RGB image to an HSI image, from the exercises'''
    # for getting no number errors 
    epsilon = 1e-5
    ## Scale the image down to the range [0,1]
    R,G,B = rgb_image[:,:,0],rgb_image[:,:,1],rgb_image[:,:,2]
    ## From equation 6-17
    # for handling right values
    arg = 0.5*((R-G)+(R-B)) / (np.sqrt((R-G)**2 + (R-B)*(G-B))+ epsilon)

    # important for having reasonabel values for arcos [-1,1]
    arccos_argument = np.where(arg < -1, -1, arg)
    arccos_argument = np.where(arg > 1, 1, arg)


    theta = np.arccos( arccos_argument)
    theta *= 180/np.pi # Convert the angle into degrees
    ## From equation 6-16
    H = np.copy(theta)
    H[B>G] = 360-theta[B>G]
    ## From equation 6-18
    S = 1-3*np.min(rgb_image, axis=2)/(np.sum(rgb_image, axis=2) + epsilon)
    ## From equation 6-19
    I = np.mean(rgb_image, axis=2)
    ## return HSI image
    hsi_image = np.stack([H,S,I], axis=2)
    return(hsi_image)

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


# exercise 3 equalizatiojn
def _hist_equalize(image):
    # takes rgb image as input and gives rgb image as output

    # Convert the image to YUV color space
    # hsi not cv implemented
    yuv_image = cv.cvtColor(image, cv.COLOR_RGB2YUV)
    
    # Apply histogram equalization to the Y channel
    yuv_image[:,:,0] = cv.equalizeHist(yuv_image[:,:,0])
    
    # Convert the image back to RGB color space
    equalized_image = cv.cvtColor(yuv_image, cv.COLOR_YUV2RGB)
    
    return(equalized_image)

# from exercises
def _hist_equalize(image):
    #### Function to perform histogram equalization
    ## Initialize the output image as a copy of the input
    new_image = np.copy(image)
    ## Find the image shape and all the unique pixel values
    M,N = image.shape
    pixel_values = np.unique(image)
    ## Initialize the probability masses of each pixel value
    p_r = []
    ## Loop through all the pixel values in the image
    for rk in pixel_values:
        ## Compute the probability mass of the current pixel value
        # and store it in the probability mass vector
        p_rk = (image==rk).sum() / (M*N)
        p_r.append(p_rk)
        ## Compute the transformed pixel value based on equation
        # (3-15) from the textbook
        s_k = 255 * sum(p_r)
        ## Replace the pixel values in the output image with the
        # transformed intensity
        new_image[image==rk] = s_k
    return(new_image)

def hist_equalize(image):
    # old and "wrong function for only rgb channels"
    new_image = np.zeros(image.shape)

    # not every channel but only the y channel !!!
    for channel in range(3):
        # prebuild functions allowed
        # int doesnt work still recognized as float
        # also tried .astype
        equalized_channel = cv.equalizeHist(image[:,:,channel])
       
        #hist, _ = np.histogram(image[:,:,channel].flatten(), 256, [0, 256])
        #cdf = np.cumsum(hist)
        #cdf_norm = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        
        #

        new_image[:,:,channel] = equalized_channel#/255
        # unnecessary
        #channel_new = cdf_norm[image[:,:,channel].flatten()]
        #new_image[:,:,channel] = np.reshape(channel_new, image[:,:,channel].shape)

    # astype int is important
    return(new_image.astype(int))




equalized_image = hist_equalize(image)
#equalized_image_box_aurora = hist_equalize(image_box_aurora)
#equalized_image_box_foreground = hist_equalize(image_box_foreground)

equalized_image_box_aurora = select_image_box(equalized_image, (260,60), (300,1000))
equalized_image_box_foreground = select_image_box(equalized_image, (630,100), (150,1000))


#plot_image_with_hists(equalized_image, "Whole image with histogram equalization.")
#plot_image_with_hists(equalized_image_box_aurora, "Aurora region with histogram equalization.")
#plot_image_with_hists(equalized_image_box_foreground, "Foreground image with histogram equalization.")



#########################################################################
#                               3                                       #
#########################################################################


hsi_image = rgb_to_hsi(image)
hsi_image_box_aurora = rgb_to_hsi(image_box_aurora)
hsi_image_box_foreground = rgb_to_hsi(image_box_foreground)

#plot_image_with_hists(hsi_image,"To hsi converted image", "HSI-Image", "H Channel", "S Channel", "I channel")
#plot_image_with_hists(hsi_image_box_aurora,"To hsi converted aurora image", "HSI-Image", "H Channel", "S Channel", "I channel")
#plot_image_with_hists(hsi_image_box_foreground,"To hsi converted foreground image", "HSI-Image", "H Channel", "S Channel", "I channel")



#########################################################################
#                               4                                       #
#########################################################################



#########################################################################
#                               5                                       #
#########################################################################

def mahalanobis_distance(z, a, c):
    #distance = (z-a).T*np.linalg.inv(c)*(z-a)
    # c is just a scalar?
    distance = (z-a).T*c*(z-a)
    return(distance)

def _image_select_mahalanobis_distance(whole_image, foreground_image, threshold):
    a_mean = np.mean(foreground_image, axis=(0, 1))

    print(a_mean.shape)
    covariance = np.cov(foreground_image.reshape(-1, 1), rowvar=False)


    distance = mahalanobis_distance(whole_image, a_mean, covariance)

    plt.imshow(distance)
    plt.show()
    
    selected_image = np.where(distance > threshold, whole_image)

    return(selected_image, distance)


def image_select_mahalanobis_distance(whole_image, foreground_image, threshold):
    # Compute the inverse of the covariance matrix
    # modification of shape .1, 1 and cov as scalar is better
    covariance = np.cov(foreground_image.reshape(-1, 3), rowvar=False)
    mean = np.mean(foreground_image, axis=(0, 1))
    print(np.amax(foreground_image)) # fix this for hsi image
    print(np.amin(foreground_image))

    

    print(foreground_image.reshape(-1, 3).shape)

    inv_covariance = np.linalg.inv(covariance)
    #inv_covariance = 1/covariance

    #print(inv_covariance) #is nan for hsi

    # Compute Mahalanobis distance for each pixel
    distance = np.zeros_like(whole_image, dtype=np.float32)
    for i in range(whole_image.shape[0]):
        for j in range(whole_image.shape[1]):
            pixel = whole_image[i, j]
            difference = pixel - mean
            # definition of distance sqrt (with/without it?)
            dist = (np.dot(np.dot(difference.T, inv_covariance), difference))
            distance[i, j] = dist


    ####


    

    ####
    print("\nRegion of distance, max/min")
    print(np.amax(distance))
    print(np.amin(distance))

    ## EROR IN DISTNACE DEFINITION???
    # Apply threshold to classify pixels as foreground or background
    #segmented_foreground = np.zeros(whole_image.shape)
    #segmented_background = np.zeros(whole_image.shape)
    print(distance.shape)
    segmented_foreground = np.where(distance < threshold, whole_image, 255)
    segmented_background = np.where(distance >= threshold, whole_image, 255)
    #segmented_foreground = whole_image[distance < threshold] #[distance < threshold]  # Foreground
    #segmented_background = whole_image[distance >= threshold]   # Background


    return(segmented_foreground, segmented_background, distance)


def plot_segmentation(segemented_foreground, segemented_background, distance, maintitle = "Image segmentation", title1 = "Segemented foreground", title2 = "Segemented background", title3 = "Distance as 2d image"):

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 7))
    fig.suptitle(maintitle)

    axs[0].set_title(title1)
    axs[0].imshow(segemented_foreground)
    axs[1].set_title(title2)
    axs[1].imshow(segemented_background)
    axs[2].set_title(title3)
    axs[2].imshow(distance, cmap = "gray")
   
    fig.tight_layout()
    plt.show()
    

    return()

#segmented_image, distance = image_select_mahalanobis_distance(image, image_box_foreground, threshold=2)

#equalized_segmented_image, equalized_distance = image_select_mahalanobis_distance(equalized_image, equalized_image_box_foreground, threshold=2)

#hsi_segmented_image, hsi_distance = image_select_mahalanobis_distance(hsi_image, hsi_image_box_foreground, threshold=2)

plot_segmentation(*image_select_mahalanobis_distance(image, image_box_foreground, threshold=550))
plot_segmentation(*image_select_mahalanobis_distance(equalized_image, equalized_image_box_foreground, threshold=80))
plot_segmentation(*image_select_mahalanobis_distance(hsi_image, hsi_image_box_foreground, threshold=8))



#########################################################################
#                               6                                       #
#########################################################################

#########################################################################
#                               7                                       #
#########################################################################



end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))