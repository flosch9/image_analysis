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
#       Task 3               Home Exam       Florian Schieren           #
#                                                                       #
#########################################################################
#########################################################################
print("\nTask 2a")
start_time = time.time()

#########################################################################
#                               1                                       #
#########################################################################
# code here

f = np.array([[2,5,7],
             [1,7,9],
             [3,3,1]])

# 180 degree flipped
g_one = np.array([[0,0,0],
                 [0,2,0],
                 [1,0,1]])

g_two = np.array([[0,1,0],
                [1,1,1],
                [0,1,0]])

f_padding_one = np.pad(f, 1)

print(f_padding_one)
print(g_one)
print(g_two)

f_pad_one_conv_g_one = signal.convolve2d(f_padding_one, g_one, mode = "valid" )

print(f_pad_one_conv_g_one)



end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))