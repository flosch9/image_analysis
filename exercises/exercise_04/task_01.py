import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#			Exercise 04			Task 1a									#
#########################################################################
print("\nTask 1a")
start_time = time.time()

from skimage.data import shepp_logan_phantom
image = shepp_logan_phantom()

n_values = np.array([400,200,50,25])

# n evenly ditributetd angles between 0 and 180 degree



fig, axs = plt.subplots(nrows=2, ncols=len(n_values)+1, figsize=(8, 7))
fig.suptitle("Radon Transformation")

axs[0,0].set_title("Original")
axs[0,0].imshow(image, cmap="gray")


for (index,n)  in enumerate(n_values):
    theta = np.linspace(0., 180., n, endpoint=False)
    sinogram = skimage.transform.radon(image, theta=theta)
    reconstruction = skimage.transform.iradon(sinogram, theta=theta)

    axs[0, index+1].set_title("n = {}".format(n))
    axs[0, index+1].imshow(sinogram, cmap = "gray")
    axs[1, index+1].set_title("Reconstruction")
    axs[1, index+1].imshow(reconstruction, cmap = "gray")

fig.tight_layout(pad=0.5)
plt.show()

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))

