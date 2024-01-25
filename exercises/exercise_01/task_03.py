import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageColor

# Task 3a
print("\n Task 3a")

picture = Image.open(r"data\DIP3E_Original_Images_CH02\Fig0207(a)(gray level band).tif")
picture.show()

# different to second method whisch allready adds some color 
# or use plt.imshow() and set cmap to grey
plt.imshow(picture)
plt.show()

picture_array = np.array(picture)
# Testing
#print(picture_array)

mid_of_picture = len(picture_array[:])/2
# better option to use may be 
#number_of_rows = np.shape(picture_array)[0]
#mid_of_picture = number_of_rows/2 

# Testing
#print(len(picture_array[:]))
#print(mid_of_picture)

x_values = np.arange(0 , len(picture_array[0][:]), step = 1)
# np.arange returns intervall [start, stop)
# take width of image as x-values
# better option again may be np.shape for number of columns

y_values = picture_array[int(mid_of_picture),:]
# take values along the mid-axis (one whole row) of image

plt.plot(x_values, y_values)
plt.xlabel("Pixel")
plt.ylabel("Intenisty")
plt.show()

unique_values_in_array = len(np.unique(picture_array))
print("Unique values in Image:", unique_values_in_array)

# Task 3b
print("\n Task 3b")

# returns a tupel, first rows second columns
shape_of_array = np.shape(picture_array)
print("Shape of array (equals Image seize):", shape_of_array)

# Task 3c
print("\n Task 3c")


"""
band_one = ImageColor.getrgb("green")
band_two = ImageColor.getrgb("red")
band_three = ImageColor.getrgb("orange")
band_four = ImageColor.getrgb("yellow")
band_five = ImageColor.getrgb("pink")
band_six = ImageColor.getrgb("black")

print(band_one)
print(np.array([band_one]))

color_bands = np.array([band_one, band_two, band_three, band_four, band_five, band_six])

number_of_rows = shape_of_array[0]
number_of_columns = shape_of_array[1]
# should have done this earlier 

grey_scale = np.unique(picture_array)



colormap = cm.get_cmap(picture_array)
print(colormap)



def change_color(pixel):
    switch_case = {
        grey_scale[0] : band_one,
        grey_scale[1] : band_two,
        grey_scale[2] : band_three,
        grey_scale[3] : band_four,
        grey_scale[4] : band_five,
        grey_scale[5] : band_six
        }    
    print(switch_case.get(pixel))              
    return(switch_case.get(pixel))
    # here use of new match feature is possible
color_picture_array = picture_array

print(grey_scale)
print(picture_array)

color_picture_array = np.where(picture_array[0] != grey_scale[0], picture_array[0], np.array([band_one]))

#for row in range(number_of_rows):
#    for column in range(number_of_columns):
#        change_color(picture_array[row, column])

print(color_picture_array)

"""

# there may be a prebuild function for getting a colormap which is better (faster?)
# maybe use matplotlib.image and use cmap argument



# Task 3d
print("\n Task 3d")



