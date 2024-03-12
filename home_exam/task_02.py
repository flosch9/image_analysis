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

image = np.asarray(Image.open(image_name).convert('L')) 
# directly convert to greysclae
#otherwise only use green channel

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

ft_image = np.fft.fftshift(np.fft.fft2(image))

ft_high, ft_width = ft_image.shape





ft_image_filtered = ft_image
ft_image_filtered[int(ft_high/2 - 10):int(ft_high/2 +10 ), int(ft_width/2 -10):int(ft_width/2 +10)] = 0

image_filtered = np.fft.ifft2(ft_image_filtered)


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("fourier filtering")

axs[0,0].set_title("Original Image")
axs[0,0].imshow(image, cmap = "gray")
axs[0,1].set_title("Fourier transformed")
axs[0,1].imshow(np.log(np.abs(ft_image)), cmap = "gray")
axs[1,0].set_title("Restoradet Image")
axs[1,0].imshow(np.log(np.abs(ft_image_filtered)), cmap = "gray")
axs[1,1].set_title("Restoradet fourier transform")
axs[1,1].imshow(image_filtered, cmap = "gray")
fig.tight_layout()
plt.show()

plt.imshow(np.log(np.abs(ft_image)), cmap = "gray")
plt.show()

#########################################################################
#                               4                                       #
#########################################################################

image_name = "home_exam\\NorthernLightNoisy2.png"

image = np.asarray(Image.open(image_name).convert('L')) 
# directly convert to greysclae
#otherwise only use green channel

ft_image = np.fft.fftshift(np.fft.fft2(image))

ft_high, ft_width = ft_image.shape()


ft_image_filtered = ft_image[int(ft_high/2 - 10):int(ft_high/2 +10 ), int(ft_width/2 -10):int(ft_width/2 +10)]


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("fourier filtering")

axs[0,0].set_title("Original Image")
axs[0,0].imshow(image, cmap = "gray")
axs[0,1].set_title("Fourier transformed")
axs[0,1].imshow(np.log(np.abs(ft_image)), cmap = "gray")
axs[1,0].set_title("Restoradet Image")
axs[1,0].imshow(np.log(np.abs(ft_image_filtered)), cmap = "gray")
axs[1,1].set_title("Restoradet fourier transform")
axs[1,1].imshow(image)
fig.tight_layout()
plt.show()

plt.imshow(np.log(np.abs(ft_image)), cmap = "gray")
plt.show()

#########################################################################
#                               5                                       #
#########################################################################


end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))