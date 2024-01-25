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

rows, columns = shape_of_array

# Task 3c
print("\n Task 3c")

grey_scale = np.unique(picture_array)

colored_picture_array = [[0]*columns]*rows

band_one = ImageColor.getrgb("green")
band_two = ImageColor.getrgb("red")
band_three = ImageColor.getrgb("orange")
band_four = ImageColor.getrgb("yellow")
band_five = ImageColor.getrgb("pink")
band_six = ImageColor.getrgb("black")

# np.array cant be used jet bcs np.aary deosnt take tupel as input
for row in range(rows):
    for column in range(columns):
        if picture_array[row][column] == grey_scale[0]:
            colored_picture_array[row][column] = band_one
        elif picture_array[row][column] == grey_scale[1]:
            colored_picture_array[row][column] = band_two
        elif picture_array[row][column] == grey_scale[2]:
            colored_picture_array[row][column] = band_three
        elif picture_array[row][column] == grey_scale[3]:
            colored_picture_array[row][column] = band_four
        elif picture_array[row][column] == grey_scale[4]:
            colored_picture_array[row][column] = band_five
        elif picture_array[row][column] == grey_scale[5]:
            colored_picture_array[row][column] = band_six
        # better use kind of switch-case  dictonary

#colored_picture_array = np.array(colored_picture_array)

plt.imshow(colored_picture_array)
plt.savefig("exercises\exercise_01\colormap.png")
plt.show()

# there may be a prebuild function for getting a colormap which is better (faster?)
# maybe use matplotlib.image and use cmap argument

# Task 3d
print("\n Task 3d")



