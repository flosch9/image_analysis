import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv
import skimage
import time
from PIL import Image 
from scipy import ndimage



#########################################################################
#			Exercise 05			Task 1a									#
#########################################################################
print("\nTask 1a")
start_time = time.time()

"""
High pass filter: supressing low frequencies and let high frequencies signasl passing
in images: "frequiencies" = edges in an image
in physical domain for example laplacian filter
in fourier domain for example inverse gaussian filter (1 - gauss) or square multiplication mask
"""

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 05			Task 1b									#
#########################################################################
print("\nTask 1b")
start_time = time.time()

"""
Filtering in spatial domain is faster than in fourier.
"""

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 05			Task 1c									#
#########################################################################
print("\nTask 1c")
start_time = time.time()

# code here

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))


#########################################################################
#			Exercise 05			Task 1d									#
#########################################################################
print("\nTask 1d")
start_time = time.time()

# code here

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))