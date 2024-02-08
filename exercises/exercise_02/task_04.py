import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image


# Task 4a
print("\n Task 4a")

def s_curve_transformation(r, r_zero, E):
    return((r/r_zero)**E/(1+((r/r_zero)**E)))

values_of_E = [2,4,6,8]
r_zero = 0.5
r = np.linspace(0,1,100)

for E_param in values_of_E:
    
    plt.plot(r, s_curve_transformation(r, r_zero, E_param), label = "E = {}, r$_0$ = {}".format(E_param, r_zero) )
    
plt.legend(loc = "best")
plt.show()


# Task 4b
print("\n Task 4b")

def s_curve_transformation_rgbrange(r, r_zero, E):
    alpha = 255/np.amax((r/r_zero)**E/(1+((r/r_zero)**E)))
    return(alpha*(r/r_zero)**E/(1+((r/r_zero)**E)))

values_of_E = [2,4,6,8]
r = np.linspace(0,255,255)
# now for range 0 to 255 adapted
r_zero_values = [50, 80, 130, 170,200]

for E_param in values_of_E:
    for r_zero in r_zero_values:
        plt.plot(r, s_curve_transformation_rgbrange(r, r_zero, E_param), label = "E = {}, r$_0$ = {}".format(E_param, r_zero) )
    
plt.legend(loc = "best", bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.7)
plt.show()

image = Image.open("data\DIP3E_Original_Images_CH03\Fig0310(b)(washed_out_pollen_image).tif")

# needs big values of r_zero because meant for range 0 to 1, 125 equals nearly the 0.5 in # range 0 to 255 like 0.5 did in range 0 to 1

fig, axes = plt.subplots(nrows=1, ncols=len(values_of_E)+1, figsize=(8, 5))
axes[0].imshow(image, cmap = "gray", vmin=0, vmax=255)
axes[0].set_title("Original")
for i, e in enumerate(values_of_E):
    axes[i+1].imshow(s_curve_transformation_rgbrange(np.array(image),r_zero = 125, E = e), cmap = "gray", vmin=0, vmax=255)
    axes[i+1].set_title("$E$ = {}".format(e))
fig.tight_layout()
plt.show()
