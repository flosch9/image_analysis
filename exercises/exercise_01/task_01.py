import numpy as np 
import pandas as pd
from scipy.ndimage import rotate

# Task 1a
print("\n Task 1a")
A = np.zeros((7,8))
print(A)

# Task 1b
print("\n Task 1b")
A[:,0] = np.ones(7)
print(A)
# solution, a bit better:
# A[:,0] = 1

A[:,1] = 2*np.ones(7)
print(A)
# solution, a bit better:
# A[:,1] = 2

A[:,2] = 3*np.ones(7)
print(A)
# solution, a bit better:
# A[:,2] = 3

A[4,6] = 5
A[5,6] = 8
A.astype(int)
print(A)

# Task 1c
print("\n Task 1c")
unique_values = np.unique(A)
print("Unique values of Matrix A", unique_values)

# Task 1d
print("\n Task 1d")
#rng = np.random.default_rng(seed=42)
# works without generator

# watch out takes shape direct, not only tuple
# in which range the numbers should be
B = np.random.rand(8,7)
print(B)
# solution, a bit better, gives random integers:
# B = np.random.randint(0,10, size=(8,7))

# Task 1e
print("\n Task 1e")
C = np.matmul(A,B)
print(C)
# solution, matrix mutiplication, a bit quicker
# C = A@B 

# Task 1f
print("\n Task 1f")
A.astype(int)
print(A)

A_flipped_updown = np.flip(A, 0) 
print(A_flipped_updown)

A_flipped_leftright = np.flip(A, 1) 
print(A_flipped_leftright)

A_rotated_ninty = np.rot90(A)
print(A_rotated_ninty)

A_rotated_thirtyseven = rotate(A, angle = 37)
print(A_rotated_thirtyseven)

# Task 1g
print("\n Task 1g")

df = pd.DataFrame(A)
df.to_csv("exercises\exercise_01\matrix_A.csv", index=False, header=False)
# solution another options to save
# np.savetxt()