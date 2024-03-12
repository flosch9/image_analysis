import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage
from scipy import signal

#########################################################################
#########################################################################
#                                                                       #
#       Task 4 - Part A      Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################

"""

print("\nTask 4 Part A")
# for measuremnt of runtime of this section
start_time = time.time()

# load data
A = np.load("home_exam\medicalimages.npz")
img_PET = A["img_PET"]

#########################################################################
#                               1                                       #
#########################################################################

# color range of the image? 0 to 1 or 0 to 255 (as assumed), np.amax gives value 3567 ???
# plot 4 pictures, at beginning, a third, two thirds and the end
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img_PET[0], cmap = "gray")
axs[0, 0].set_title("Image at Postiton {} from {}".format(0, len(img_PET)))
axs[0, 1].imshow(img_PET[int(len(img_PET)/3)], cmap = "gray")
axs[0, 1].set_title("Image at Postiton {} from {}".format(int(len(img_PET)/3), len(img_PET)))
axs[1, 0].imshow(img_PET[int(2*len(img_PET)/3)], cmap = "gray")
axs[1, 0].set_title("Image at Postiton {} from {}".format(int(2*len(img_PET)/3), len(img_PET)))
axs[1, 1].imshow(img_PET[-1], cmap = "gray")
axs[1, 1].set_title("Image at Postiton {} from {}".format(len(img_PET), len(img_PET)))

fig.tight_layout()

#plt.show()
plt.close()


#########################################################################
#                               2                                       #
#########################################################################


plt.imshow(np.amax(img_PET, axis = 1), cmap = "gray")
#plt.show()
plt.close()


#########################################################################
#                               3                                       #
#########################################################################

# get histogram (build in via cv or use numpy)

#TODO here

plt.imshow(np.amax(img_PET, axis = 1), cmap = "gray", vmax=np.amax(img_PET)/3)
#plt.show()
plt.imshow(np.amax(img_PET, axis = 2), cmap = "gray", vmax=np.amax(img_PET)/2)
#plt.show()
plt.close()

# measuremnt of runtime of this section
end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#########################################################################
#                                                                       #
#       Task 4 - Part B      Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################

print("\nTask 4 Part B")
# for measuremnt of runtime of this section
start_time = time.time()

# load data
A = np.load("home_exam\medicalimages.npz")
sinogram = A["sinogram"]

#########################################################################
#                               1                                       #
#########################################################################
# TODO label axis
plt.imshow(sinogram, cmap = "gray")
#plt.show()


#########################################################################
#                               2                                       #
#########################################################################
# TODO label axis

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 7))
fig.suptitle("Backtronsformation noise")

axs[0].set_title("Original Backtransformation")
axs[0].imshow(skimage.transform.iradon(sinogram), cmap = "gray")
axs[1].set_title("No filter")
axs[1].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20), filter_name = None), cmap = "gray")
axs[2].set_title("Ramp filter")
axs[2].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20), filter_name = "ramp"), cmap = "gray")
fig.tight_layout()
#plt.show()
plt.close()

#########################################################################
#                               3                                       #
#########################################################################
#TODO

#########################################################################
#                               4                                       #
#########################################################################

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 7))
fig.suptitle("With noise")

axs[0].set_title("No filter")
axs[0].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = None), cmap = "gray")
axs[1].set_title("Ramp filter")
axs[1].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = "ramp"), cmap = "gray")
axs[2].set_title("Cosine filter")
axs[2].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = "cosine"), cmap = "gray")
fig.tight_layout()
#plt.show()
plt.close()


#########################################################################
#                               5                                       #
#########################################################################
# TODO


#########################################################################
#                               6                                       #
#########################################################################
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 7))
fig.suptitle("Selected area")

#TODO better code

axs[0].set_title("No filter")
axs[0].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = None)[200:400,50:200], cmap = "gray")
axs[1].set_title("Ramp filter")
axs[1].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = "ramp")[200:400,50:200], cmap = "gray")
axs[2].set_title("Cosine filter")
axs[2].imshow(skimage.transform.iradon(sinogram+np.random.rayleigh(scale = 20, size = sinogram.shape), filter_name = "cosine")[200:400,50:200], cmap = "gray")
fig.tight_layout()
#plt.show()
plt.close()


# measuremnt of runtime of this section
end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))

"""

#########################################################################
#########################################################################
#                                                                       #
#       Task 4 - Part C      Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################

print("\nTask 4 Part C")
# for measuremnt of runtime of this section
start_time = time.time()

# load data
A = np.load("home_exam\medicalimages.npz")

img_dPET = A["img_dPET"]

#########################################################################
#                               1                                       #
#########################################################################

print(img_dPET.shape)

#TODO check if right timestep
plt.imshow(np.amax(img_dPET[0,:,:,:], axis = 1), cmap = "gray")
plt.show()

#########################################################################
#                               2                                       #
#########################################################################

def mean_filter_3d(cube_3d, kernelsize):
    #filtered_3dcube = np.zeros(cube_3d.shape)
    kernel = np.ones((kernelsize,kernelsize,kernelsize))/(kernelsize**3)
    
    filtered_3dcube = signal.convolve(cube_3d, kernel, method = "direct")
    # following only works with one d arrays
    #filtered_3dcube = np.convolve(cube_3d, kernel, mode = "valid")


    return(filtered_3dcube)


plt.imshow(np.amax(mean_filter_3d(img_dPET[2,:,:,:], 7), axis =1), cmap = "gray")
plt.show()

#########################################################################
#                               3                                       #
#########################################################################



# measuremnt of runtime of this section
end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))