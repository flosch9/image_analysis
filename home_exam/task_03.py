import numpy as np
import time
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

# define the input matrix
f = np.array([[2,5,7],
              [1,7,9],
              [3,3,1]])

# define the filter g_1
g_one = np.array([[1,0,1],
                  [0,2,0],
                  [0,0,0]])

# define the filter g_1
g_two = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])

# flipping the filters at the horizontal axis
# since filter g_two is symmetric on both axis, 
# flipping has no impact on this filter
g_one_flipped = np.flip(g_one, axis = 0)
g_two_flipped = np.flip(g_two, axis = 0)

# calculating the convolution and corelation for different paddings 
for pad_width in [1,2]:
    print("\nPadding of {}".format(pad_width))

    # pad the input matrix
    padded_f = np.pad(f, pad_width)

    print("Padded input f:\n")
    print(padded_f)
    
    # calculate convolution and correlation
    # in mode "valid" the output consists only of those elements that 
    # do not rely on the zero-padding
    convolution_g_one = signal.convolve2d(padded_f, g_one, mode = "valid")
    convolution_g_two = signal.convolve2d(padded_f, g_two, mode = "valid")

    correlation_g_one = signal.correlate2d(padded_f, g_one, mode = "valid")
    correlation_g_two = signal.correlate2d(padded_f, g_two, mode = "valid")

    # printing results
    print("\nConvolution with g_1:\n")
    print(convolution_g_one)
    print("\nConvolution with g_2:\n")
    print(convolution_g_two)
    print("\nCorrelation with g_1:\n")
    print(correlation_g_one)
    print("\nCorrelation with g_2:\n")
    print(correlation_g_two)

    # do the same as before but with flipped filters to show that convolution 
    # and correlation are the same for a flipped (180 degree rotated) filter
    print("\nNow the same but with flipped filters (rotation of 180 degree) :")

    convolution_g_one_flipped = signal.convolve2d(padded_f, g_one_flipped, mode = "valid" )
    convolution_g_two_flipped = signal.convolve2d(padded_f, g_two_flipped, mode = "valid" )

    correlation_g_one_flipped = signal.correlate2d(padded_f, g_one_flipped, mode = "valid" )
    correlation_g_two_flipped = signal.correlate2d(padded_f, g_two_flipped, mode = "valid" )

    print("\nConvolution with flipped g_1:\n")
    print(convolution_g_one_flipped)
    print("\nConvolution with flipped g_2:\n")
    print(convolution_g_two_flipped)
    print("\nCorrelation with flipped g_1:\n")
    print(correlation_g_one_flipped)
    print("\nCorrelation with flipped g_2:\n")
    print(correlation_g_two_flipped)

    """
    # check if results for convolution/ correlation are really the same 
    # for flipped/ not flipped filter
    print("\nComparison of convolution and correlation with flipped kernel:\n")
    print(correlation_g_one == convolution_g_one_flipped)
    print(convolution_g_one == correlation_g_one_flipped)
    """

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))