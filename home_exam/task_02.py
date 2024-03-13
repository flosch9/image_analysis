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
#       Task 2               Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################
print("\nTask 2a")
start_time = time.time()

#########################################################################
#                               1                                       #
#########################################################################
# TODO

#########################################################################
#                               2                                       #
#########################################################################
# TODO

#########################################################################
#                               3                                       #
#########################################################################

image_name = "home_exam\\NorthernLightNoisy1.png"

#image = np.asarray(Image.open(image_name).convert('L')) 
# directly convert to greysclae
image = np.asarray(Image.open(image_name))
#otherwise only use green channel



def postprocess(gxy, fxy):
    ## Posptrocessing g(x,y)
    ## Isolate only the real component of the image
    g = np.real(gxy)
    ## Remove the zero-padded edges of the image
    target_shape = fxy.shape
    dim_dif = [g.shape[i]-target_shape[i] for i in range(g.ndim)]
    g = g[dim_dif[0]//2:-dim_dif[0]//2 , dim_dif[1]//2:-dim_dif[1]//2]
    return g

def preprocess(fxy):
    ## Preprocess f(x,y)
    ## Determine how much to zero-pad the image
    # We are determining how many zeros to pad each edge with to make the
    # image dimensions a power of 2.
    shape = fxy.shape[:2]
    pad_dims = []
    for dim in shape:
        new_dim = 2**np.ceil(np.log2(dim))
        pad_val = int((new_dim - dim)//2)
        pad_dims.append( (pad_val, pad_val) )
    ## Zero-pad the image
    _fxy = np.pad(fxy, pad_dims, mode='constant', constant_values=0)
    return _fxy

def ft_filtering():
    return
"""
# preview fourier domain
for channel in range(image.shape[2]):
    ft = np.fft.fftshift(np.fft.fft2(preprocess(image[:,:,channel])))
    plt.imshow(np.log(np.abs(ft)), cmap = "gray")
    plt.show()

"""

filters = np.ones(image.shape)

image_filtered = np.copy(image)

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 7))
fig.suptitle("fourier filtering")

axs[0,0].set_title("Original Image")
axs[0,0].imshow(image)


for channel in range(image.shape[2]):
    image_channel = image[:,:,channel]

    ft_channel = np.fft.fftshift(np.fft.fft2(preprocess(image_channel)))

    filters = np.ones((ft_channel.shape[0], ft_channel.shape[1],3))
    

    # r filters
    #filters[511:513,:490,0] = 0
    #filters[511:513,520:,0] = 0
    # g filter
    filters[482:493, :,1] = 0
    filters[532:542, :,1] = 0

    # b filters
    #filters[511:513,:490,2] = 0
    #filters[511:513,520:,2] = 0

    ft_channel_filtered = ft_channel * filters[:,:,channel]

    back_ft = np.fft.ifft2(np.fft.fftshift(ft_channel_filtered))

    image_filtered[:,:,channel]=postprocess(back_ft, image_channel)

    axs[channel+1,0].set_title("Fourier transformed")
    axs[channel+1,0].imshow(np.log(np.abs(ft_channel)), cmap = "gray")
    axs[channel+1,1].set_title("Filtered fourier image")
    axs[channel+1,1].imshow(np.log(np.abs(ft_channel_filtered)), cmap = "gray")
    
axs[0,1].set_title("Restored image")
axs[0,1].imshow(image_filtered)
fig.tight_layout()
#plt.show()
plt.close()

# noise in blue channel?

# show final result
plt.imshow(image_filtered)
#plt.show()
plt.close()

# remove salt noise
# kernelsize should be an odd number
kernelsize = 3
image_final_result_1 = cv.medianBlur(image_filtered, kernelsize)

plt.imshow(image_final_result_1)
plt.show()
plt.close()


#########################################################################
#                               4                                       #
#########################################################################

image_name = "home_exam\\NorthernLightNoisy2.png"

image = np.asarray(Image.open(image_name)) 
# directly convert to greysclae
#otherwise only use green channel

print(image.shape)

"""
# preview fourier domain
for channel in range(image.shape[2]):
    ft = np.fft.fftshift(np.fft.fft2(preprocess(image[:,:,channel])))
    plt.imshow(np.log(np.abs(ft)), cmap = "gray")
    plt.show()

"""

def contraharmonic_mean_filter(image, order, filtersize = 3):


    filtered_image = 1

    return(filtered_image)

def median_filter(image, kernelsize):
    filtered_image = cv.medianBlur(image, kernelsize)
    return(filtered_image)

filters = np.ones(image.shape)

image_filtered = np.copy(image)

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 7))
fig.suptitle("fourier filtering")

axs[0,0].set_title("Original Image")
axs[0,0].imshow(image)


for channel in range(image.shape[2]):
    image_channel = image[:,:,channel]

    ft_channel = np.fft.fftshift(np.fft.fft2(preprocess(image_channel)))

    filters = np.ones((ft_channel.shape[0], ft_channel.shape[1],3))
    

    # r filters
    filters[515:520,485:507,:] = 0
    filters[503:507,517:540,:] = 0
    # g filter
    #filters[503:506, :,1] = 0
    #filters[503:506, :,1] = 0

    # b filters
    #filters[503:506,:490,2] = 0
    #filters[503:506,520:,2] = 0

    ft_channel_filtered = ft_channel * filters[:,:,channel]

    back_ft = np.fft.ifft2(np.fft.fftshift(ft_channel_filtered))

    image_filtered[:,:,channel]=postprocess(back_ft, image_channel)

    axs[channel+1,0].set_title("Fourier transformed")
    axs[channel+1,0].imshow(np.log(np.abs(ft_channel)), cmap = "gray")
    axs[channel+1,1].set_title("Filtered fourier image")
    axs[channel+1,1].imshow(np.log(np.abs(ft_channel_filtered)), cmap = "gray")
    
# removed peridoic noise



axs[0,1].set_title("Image with removed periodic noise")
axs[0,1].imshow(image_filtered)
fig.tight_layout()
#plt.show()
plt.close()

# remove salt noise
# better blur in ft domain with low pass filter!
kernelsize = 5
image_final_result_2 = cv.medianBlur(image_filtered, kernelsize)

plt.imshow(image_final_result_2)
#plt.show()
plt.close()

#########################################################################
#                               5                                       #
#########################################################################

# image one
print("\nEnhancement of image 1:")

"""
#doesnt work method 
kernelsize = (5,5)
blurred_version =  cv.blur(image_final_result_1, kernelsize)
enhanced_image = blurred_version - image_final_result_1 
"""
# sharpening with filter
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])

enhanced_image = cv.filter2D(image_final_result_1, -1, kernel)

plt.imshow(enhanced_image)
plt.show()

#also gaus blur and mask subtracted add to image possiple

"""
alpha = 1.1
beta = 20

more_enhanced_image = cv.addWeighted(enhanced_image, alpha, enhanced_image, 0, beta)
plt.imshow(more_enhanced_image)
plt.show()

"""

"""
# even worse
image_hsv = cv.cvtColor(enhanced_image, cv.COLOR_RGB2HSV) 
  
# Adjust the hue, saturation, and value of the image 
# Adjusts the hue by multiplying it by 0.7 
image_hsv[:, :, 0] = image_hsv[:, :, 0] * 1.3
# Adjusts the saturation by multiplying it by 1.5 
image_hsv[:, :, 1] = image_hsv[:, :, 1] * 0.98
# Adjusts the value by multiplying it by 0.5 
image_hsv[:, :, 2] = image_hsv[:, :, 2] * 1
  
# Convert the image back to BGR color space 
more_enhanced_image = cv.cvtColor(image_hsv, cv.COLOR_HSV2BGR) 

plt.imshow(more_enhanced_image)
plt.show()

"""


# image two


print("\nEnhancement of image 2:")


def sharpen_ft_space(channel):
    # fourier transform
    ft_channel = np.fft.fftshift(np.fft.fft2(preprocess(channel)))
    # filter (high pass filter!)
    rows, cols = ft_channel.shape
    mask = np.ones((rows, cols))
    radius = 60
    mask[int(rows/2 - radius):int(rows/2 + radius), int(cols/2 - radius):int(cols/2 + radius)] = 0
    # apply filter
    ft_channel_sharpened = ft_channel * mask
    # back transform
    sharpened_channel = np.fft.ifft2(np.fft.ifftshift(ft_channel_sharpened))
    #sharpened_channel = postprocess(sharpened_channel, channel)

    return(postprocess(sharpened_channel, channel))

enhanced_image = np.copy(image_final_result_2)

for channel in range(image_final_result_2.shape[2]):
    image_channel = image_final_result_2[:,:,channel]

    sharpened_channel = sharpen_ft_space(image_channel)

    enhanced_image[:,:,channel] = sharpened_channel

plt.imshow(image_final_result_2)
plt.show()

plt.imshow(enhanced_image)
plt.show()


end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))