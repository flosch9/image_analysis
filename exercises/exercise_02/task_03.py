import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def gamma_transformation(array_of_gamma, plot_gammatransformation = True, adabt_range = False, image = None):

    if image == None:
        r = np.arange(0, 256, step = 1)
    else:
        r = np.array(image)
        
    return_image = []

    for gamma in array_of_gamma:

        if adabt_range == True:
            #c = 255/np.amax(r**gamma)
            #error here: c = 255**(1-gamma) !!!
            c = 255**(1-gamma)
        elif adabt_range == False:
            c = 1

        s = c*r**gamma
        return_image.append(s)
        
        print("\nGammavalue: {}".format(gamma))
        print("Range for r: ({}, {})".format(np.amin(r), np.amax(r)))
        print("Range for s: ({}, {})".format(np.amin(s), np.amax(s)))

        plt.plot(r, s, label = "$\gamma$ = {}".format(gamma))

    if plot_gammatransformation == True: 
        plt.legend(loc = "best")       
        plt.show()
    else:
        plt.close()
        
    if image != None:
        return(return_image)
    
    pass

# Task 3a
print("\n Task 3a")

gamma = [0.6,0.4,0.3,0.1]

gamma_transformation(gamma, plot_gammatransformation = True, adabt_range = False)

image = Image.open("data\DIP3E_Original_Images_CH03\Fig0308(a)(fractured_spine).tif")

transformed_image = gamma_transformation(gamma, plot_gammatransformation = False, adabt_range = False, image = image)

fig, axes = plt.subplots(nrows=1, ncols=len(gamma)+1, figsize=(8, 5))
axes[0].imshow(image, cmap = "gray", vmin=0, vmax=255)
axes[0].set_title("Original")
for i, gamma in enumerate(gamma):
    axes[i+1].imshow(transformed_image[i], cmap = "gray", vmin=0, vmax=255)
    axes[i+1].set_title("$\gamma$ = {}".format(gamma))
fig.tight_layout()
plt.show()


# Task 3b
print("\n Task 3b")

gamma = [0.6,0.4,0.3]

gamma_transformation(gamma, plot_gammatransformation = True, adabt_range = True)

transformed_image = gamma_transformation(gamma, plot_gammatransformation = False, adabt_range = True, image = image)

fig, axes = plt.subplots(nrows=1, ncols=len(gamma)+1, figsize=(8, 5))
axes[0].imshow(image, cmap = "gray", vmin=0, vmax=255)
axes[0].set_title("Original")
for i, gamma in enumerate(gamma):
    axes[i+1].imshow(transformed_image[i], cmap = "gray", vmin=0, vmax=255)
    axes[i+1].set_title("$\gamma$ = {}".format(gamma))

fig.tight_layout()
plt.show()
