import numpy as np
import matplotlib.pyplot as plt 


# Task 4a
print("\n Task 4a")

def s_curve_transformation(r, r_zero, E):
    return((r/r_zero)**E/(1+(r/r_zero**E)))

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
    alpha = 255/max((r/r_zero)**E/(1+(r/r_zero**E)))
    return(alpha*(r/r_zero)**E/(1+(r/r_zero**E)))

r = np.linspace(0,255,255)
r_zero_values = [0.2,0.7,1.6,5]


for E_param in values_of_E:
    for r_zero in r_zero_values:
        plt.plot(r, s_curve_transformation_rgbrange(r, r_zero, E_param), label = "E = {}, r$_0$ = {}".format(E_param, r_zero) )
    
plt.legend(loc = "best", bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.7)
plt.show()

