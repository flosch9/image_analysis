import numpy as np
import matplotlib.pyplot as plt 

def s_curve_transformation(r, r_zero, E):
    return((r/r_zero)**E/(1+(r/r_zero**E)))

# Task 4a
print("\n Task 4a")

values_of_E = [2,4,6,8]
r_zero = 0.5
r = np.linspace(0,1,100)

for E_param in values_of_E:
    plt.plot(r, s_curve_transformation(r, r_zero, E_param))

plt.show()


# Task 4b
print("\n Task 4b")