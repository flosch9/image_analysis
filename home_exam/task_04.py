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
#       Task 4 - Part A      Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################

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
plt.show()
plt.imshow(np.amax(img_PET, axis = 2), cmap = "gray", vmax=np.amax(img_PET)/2)
plt.show()


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




# measuremnt of runtime of this section
end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))

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



# measuremnt of runtime of this section
end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))