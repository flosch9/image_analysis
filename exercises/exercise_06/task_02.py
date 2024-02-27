import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#			Exercise 06			Task 2a									#
#########################################################################
print("\nTask 2a")
start_time = time.time()

image_name = "data\\blaklokke.jpg"

image = Image.open(image_name)

# important for slicing later
image = np.asarray(image)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("Spatial Filters")

#print(image.shape)

axs[0,0].set_title("Original blaklokke")
axs[0,0].imshow(image)
axs[0,1].set_title("R blaklokke")
axs[0,1].imshow(image[:,:,0], cmap="gray", vmin = 0, vmax = 255)
axs[1,0].set_title("G blaklokke")
axs[1,0].imshow(image[:,:,1], cmap="gray", vmin = 0, vmax = 255)
axs[1,1].set_title("B blaklokke")
axs[1,1].imshow(image[:,:,2], cmap="gray", vmin = 0, vmax = 255)

plt.show()
#rgb -> b channel is the best suitable for detecting the color

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 06			Task 2b									#
#########################################################################
print("\nTask 2b")
start_time = time.time()

# same image from task 1a

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("Spatial Filters")

axs[0,0].set_title("Original blaklokke / overall")
axs[0,0].hist(image.flatten(), 255)
axs[0,1].set_title("R blaklokke / red channel")
axs[0,1].hist(image[:,:,0].flatten(), 255)
axs[1,0].set_title("G blaklokke / green channel")
axs[1,0].hist(image[:,:,1].flatten(), 255)
axs[1,1].set_title("B blaklokke / blue channel")
axs[1,1].hist(image[:,:,2].flatten(), 255)

plt.show()

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))
#########################################################################
#			Exercise 06			Task 2c									#
#########################################################################
print("\nTask 2c")
start_time = time.time()

def rgb_to_hsi(rgb_image):
    
    rows, columns, _, = rgb_image.shape

    hsi_image = np.zeros((rows, columns, 3))

    # slow, works also without a for loop
    # hsi cone like shape imagine
    for row in range(rows):
        for column in range(columns):
            r,g,b = rgb_image[row,column,:]/255 # represent in range 0 to 1
            
            theta = np.arccos((0.5*((r-g)+(r-b)))/(np.sqrt((r-g)**2 + (r-b)*(g-b)) + 1e-8)) #produces a warning overflow/ dividing by zero
            # lasz term with 1e-8 for not getting a warning

            # theta has to be in degree (nor radians)
            theta *=180/np.pi

            hue = theta if b<=g else 360-theta

            int = (r+g+b)/3

            sat = 1 - (np.min([r,g,b]))

            

            hsi_image[row, column,:] = [hue, sat, int]

    return(hsi_image)

hsi_image = rgb_to_hsi(image)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("Spatial Filters")

#print(image.shape)

axs[0,0].set_title("Original blaklokke")
axs[0,0].imshow(hsi_image)
axs[0,1].set_title("H blaklokke")
axs[0,1].imshow(hsi_image[:,:,0], cmap="gray", vmin = 0, vmax = 255)
axs[1,0].set_title("S blaklokke")
axs[1,0].imshow(hsi_image[:,:,1], cmap="gray", vmin = 0, vmax = 255)
axs[1,1].set_title("I blaklokke")
axs[1,1].imshow(hsi_image[:,:,2], cmap="gray", vmin = 0, vmax = 255)

plt.show()


end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 06			Task 2d									#
#########################################################################
print("\nTask 2d")
start_time = time.time()

# same hsi image from above

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
fig.suptitle("Spatial Filters")

axs[0,0].set_title("Original blaklokke / overall")
axs[0,0].hist(hsi_image.flatten(), 255)
axs[0,1].set_title("H blaklokke / red channel")
axs[0,1].hist(hsi_image[:,:,0].flatten(), 255)
axs[1,0].set_title("S blaklokke / green channel")
axs[1,0].hist(hsi_image[:,:,1].flatten(), 255)
axs[1,1].set_title("I blaklokke / blue channel")
axs[1,1].hist(hsi_image[:,:,2].flatten(), 255)

plt.show()

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))