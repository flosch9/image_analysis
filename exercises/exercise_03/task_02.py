import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#			Exercise 03			Task 2a									#
#########################################################################
print("\nTask 2a")
start_time = time.time()

images = ["image_analysis_codes\data\DIP3E_Original_Images_CH03\Fig0316(1)(top_left).tif",
          "image_analysis_codes\data\DIP3E_Original_Images_CH03\Fig0316(2)(2nd_from_top).tif",
          "image_analysis_codes\data\DIP3E_Original_Images_CH03\Fig0316(3)(third_from_top).tif",
          "image_analysis_codes\data\DIP3E_Original_Images_CH03\Fig0316(4)(bottom_left).tif"]

fig, axs = plt.subplots(nrows=len(images), ncols=2, figsize=(7, 6))
fig.suptitle("Histograms Images 03.16")


for index, image_name in enumerate(images):
    
    image = Image.open(image_name)

    hist_data = image.histogram()
    x_values = np.arange(0, len(hist_data), 1)
    axs[index,0].bar(x_values, hist_data)
    axs[index,1].imshow(image, cmap="gray", vmin = 0, vmax = 255)

plt.show()

end_time = time.time()
print("Completetd in {}s.".format(start_time-end_time))


#########################################################################
#			Exercise 03			Task 2b									#
#########################################################################
print("\nTask 2b")
start_time = time.time()

# code here

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))