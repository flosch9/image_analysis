import numpy as np
import matplotlib.pyplot as plt 

def gamma_transformation(array_of_gamma, plot_gammatransformation = True, adabt_range = False, image = None):
    r = np.arange(0, 256, step = 1)
    s = np.arange(0, 256, step = 1)

    if adabt_range == True:
        c = 1
        print("Range with 'correction' of range")
    elif adabt_range == False:
        c = 1
        print("Range without 'correction' of range")

    
    for gamma in array_of_gamma:
        s = c*r**gamma
        
        print("Range for r: ({}, {})".format(min(r), max(r)))
        print("Range for r: ({}, {})".format(min(s), max(s)))

        s_plot = 256/max(s) * r**gamma
        if adabt_range == True:
            s_plot = 256/max(s) * r**gamma
            plt.plot(r, s_plot)
        else:
            plt.plot(r, s_plot)


    if plot_gammatransformation == True:        
        plt.show()

    pass

# Task 3a
print("\n Task 3a")

gamma_transformation([0.6,0.4,0.3, 1, 1.5, 0.05], plot_gammatransformation = True, adabt_range = False)

# Task 3a
print("\n Task 3a")

gamma_transformation([0.6,0.4,0.3])