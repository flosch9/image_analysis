import numpy as np
import matplotlib.pyplot as plt 

r = np.linspace(0,1,100)
# for x values in range of intervall [0,1]

# Task 6a
print("\n Task 6a")

A_param = 1
L_zero = 0.2

def transformation_a(r, A = 1, L_zero = 1):
    alpha = - np.log(1/2)/L_zero**2
    return(A * np.exp(-alpha*r**2))

plt.plot(r, transformation_a(r, A_param, L_zero), label = "$A \cdot \exp(-\\alpha \cdot r^2)$")
plt.hlines(A_param/2, 0, L_zero, linestyles = "dashed", color = "red", label = "A/2 with A = {}".format(A_param))
plt.vlines(L_zero, 0, A_param/2, linestyles = "dashed", color = "black")
plt.legend(loc = "best")
plt.show()

# Task 6b
print("\n Task 6b")

B_param = 5
L_zero = 0.1

def transformation_b(r, B_param = 1, L_zero = 1):
    # hard to solve with boundary conditions 
    # f(1) = B and
    # f(0) = 0
    # try instead 
    # f(0) = 0
    # f(inf) -> B
    alpha = - np.log(1/2)/L_zero**2
    c = -B_param
    return(c * (np.exp(- alpha*r**2)-1))

plt.plot(r, transformation_b(r, B_param, L_zero), label = "$- B \cdot (\exp(-\\alpha \cdot r^2) - 1)$")
plt.hlines(B_param/2, 0, L_zero, linestyles = "dashed", color = "red", label = "B/2 with B = {}".format(B_param))
plt.vlines(L_zero, 0, B_param/2, linestyles = "dashed", color = "black")
plt.legend(loc = "best")
plt.show()

# Task 6c
print("\n Task 6c")

D_param = 2
C_param = 1
L_zero = 0.3
# here L_zero is not needed to give as paramater but it is easier to use it 
# for the right range to show the plots and to determine the alpha 
# a value <0.5 for L_zero leads to a plot may not showwing the whole curve 

def transformation_c(r, c_param, d_param, L_zero = 0.3):

    if c_param >= d_param:
        print("Error. It should be D > C")
        return()
    
    alpha = - np.log(1/2)/L_zero**2
    beta = c_param - d_param
    gamma = d_param
    return(beta * (np.exp(- alpha*r**2)) + gamma)


plt.plot(r, transformation_c(r, C_param, D_param), label = "$(C-D) \cdot \exp(-\\alpha \cdot r^2) + D$")
plt.hlines(D_param, 0, 1, linestyles = "dashed", color = "black", label = "D = {}".format(D_param))
plt.hlines(C_param, 0, 1, linestyles = "dashed", color = "red", label = "C = {}".format(C_param))
plt.ylim(ymin=0) 
plt.legend(loc = "best")
plt.show()