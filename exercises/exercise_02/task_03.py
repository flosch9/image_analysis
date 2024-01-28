import numpy as np
import matplotlib.pyplot as plt 

def gamma_transformation(array_of_gamma, plot_gammatransformation = True, adabt_range = False, image = None):

    if image == None:
        r = np.arange(0, 256, step = 1)

    for gamma in array_of_gamma:

        if adabt_range == True:
            c = 255/max(r**gamma)
        elif adabt_range == False:
            c = 1

        s = c*r**gamma
        
        print("\nGammavalue: {}".format(gamma))
        print("Range for r: ({}, {})".format(min(r), max(r)))
        print("Range for s: ({}, {})".format(min(s), max(s)))

        plt.plot(r, s, label = "$\gamma$ = {}".format(gamma))

    if plot_gammatransformation == True: 
        plt.legend(loc = "best")       
        plt.show()

    pass

# Task 3a
print("\n Task 3a")

gamma_transformation([0.6,0.4,0.3, 1, 1.5, 0.05], plot_gammatransformation = True, adabt_range = False)

# Task 3a
print("\n Task 3a")

gamma_transformation([0.6,0.4,0.3, 1, 1.5, 0.05], plot_gammatransformation = True, adabt_range = True)