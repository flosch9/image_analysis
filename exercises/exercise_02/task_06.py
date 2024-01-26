import numpy as np
import matplotlib.pyplot as plt 

r = np.linspace(0,1,100)

def transformation_a(r, A = 1, L_zero = 1):
    alpha = - np.log(1/2)/L_zero**2
    return(A * np.exp(-alpha*r**2))

A_param = 1
L_zero = 0.2

plt.plot(r, transformation_a(r, A_param, L_zero))
plt.hlines(A_param/2, 0, L_zero, linestyles = "dashed")
plt.vlines(L_zero, 0, A_param/2, linestyles = "dashed")
plt.show()


B_param = 1
L_zero = 0.2

def transformation_b(r, B = 1, L_zero = 1):
    alpha = - np.log(1/2)/np.sqrt(L_zero)
    return(B * (1 - np.exp(-alpha*r**2)))


plt.hlines(B_param/2, 0, L_zero, linestyles = "dashed")
plt.vlines(L_zero, 0, B_param/2, linestyles = "dashed")
plt.plot(r, transformation_b(r, B_param, L_zero))
plt.show()


def transformation_c(r):

    return(r)


plt.plot(r, transformation_c(r))
plt.show()