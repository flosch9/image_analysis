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
plt.show()


plt.imshow(image_filtered)
plt.show()

"""
#########################################################################
#                               4                                       #
#########################################################################

image_name = "home_exam\\NorthernLightNoisy2.png"

image = np.asarray(Image.open(image_name)) 
# directly convert to greysclae
#otherwise only use green channel

print(image.shape)

# preview fourier domain
for channel in range(image.shape[2]):
    ft = np.fft.fftshift(np.fft.fft2(preprocess(image[:,:,channel])))
    plt.imshow(np.log(np.abs(ft)), cmap = "gray")
    plt.show()



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
    filters[503:506,475:510,:] = 0
    filters[503:506,518:550,:] = 0
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
    
axs[0,1].set_title("Restored image")
axs[0,1].imshow(image_filtered)
fig.tight_layout()
plt.show()


plt.imshow(image_filtered)
plt.show()

#########################################################################
#                               5                                       #
#########################################################################


end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))