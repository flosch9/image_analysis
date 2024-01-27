# Task 5
print("\n Task 5")

# define a linear function s = m*r+c which stretches the initial intenisties r
# to the range 0 -> L-1 where L is the max Intenisty of the initial intensities
# the slope m of the function is given with m = (L-1)/(L -  r_0) where
# r_0 is the minimum of the initial intensities 

def stretch_transformation(array_of_intensity_values):

    max_intensity = max(array_of_intensity_values)
    min_intensity = min(array_of_intensity_values)
    range_intensity = max_intensity - min_intensity

    slope = (max_intensity - 1) / range_intensity

    s = array_of_intensity_values * slope - min_intensity
    return(s)