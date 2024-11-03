import numpy as np
import matplotlib.pyplot as plt

rand_num = np.random.randint(0, 10) # [0, 10)
print("RAND NUM: ",rand_num)
rand_float = np.random.rand()
print("RAND FLOAT: ", rand_float) # between 0 and 1
rand_arr = np.random.randint(0, 15, 5)
print("Random Array: ", rand_arr)
rand_f_arr = np.random.rand(5)
print(rand_f_arr)
rand_2d = np.random.randint(0, 100, (3, 4))
print("RANDOM 2d: ")
print(rand_2d)
# Choose random number from array
num = np.random.choice(rand_arr) # Must be single D Array
print("RANDOM CHOICE: ", num)

a1 = np.array([1, 3, 5])
a2 = np.array([2, 4, 6])
print(np.outer(a1, a2))
'''
The outer() function in NumPy computes the outer product of two
 arrays, which is the product of all possible pairs of their 
 entries.
1*2      1*4       1*6        
3*2      3*4       3*6
5*2      5*4       5*6
'''
# NOTE: IMPORTANT solve() function
'''
NumPy solve() Function
In NumPy, we use the solve() function to solve a system of linear equations.

For a given matrix A and a vector b, solve(A, b) finds the solution vector x that satisfies the equation Ax = b.
'''
A = np.array([[2, 4],[6, 8]])
b = np.array([5, 6])
x = np.linalg.solve(A, b)
#Ax=b
print("SOLUTION: ",x)
'''
In this example, we have used the solve() function from the linalg module to solve the system of linear equations.

Here, the output is [-2. 2.25], which is the solution to the system of linear equations 2x + 4y = 5 and 6x + 8y = 6.
'''
'''
Note: The solve() function only works for square matrices, and it assumes that the matrix A has a non-zero determinant, else solve() will raise a LinAlgError exception.
'''
print("INVERSE: ")
print(np.linalg.inv(A))
 # TRACE() Function used to find the sum of diagonal elements
arr2d = np.random.randint(0, 11, (3, 3))
print(arr2d)
print("TRACE of above matrix: ",np.trace(arr2d))

# Histogram and plot
'''
The histogram() function returns a tuple containing two arrays:

the first array contains the frequency counts of the data within each bin, and
the second array contains the bin edges.
From the resulting output, we can see that:

Only 1 data point (i.e., 5) from the array data lies between the bin edges 0 and 10
3 data points (i.e., 10, 15, 18) lie between 10 and 20, and
1 data point (i.e., 20) lies between 20 and 30.
'''
data = np.array([1, 2, 3, 4, 5, 6, 3, 2, 1, 3, 4, 6, 8, 12, 13, 14, 12, 11])
bin = np.array([0, 5, 10, 15])

graph = np.histogram(data, bin)
print(graph)
plt.hist(data, bin)
# plt.show()

# INTERPOLATION
day = np.array([2, 4, 7])
gold_price = np.array([55, 58, 57])
day3_val = np.interp(3, day, gold_price)
print("Gold price on day 3: ",day3_val)
'''
Here, we used the interp() function to find the value of gold on day 3.

We have sent 3 arguments to the interp() function:

3: coordinate whose value needs to be interpolated
day: x-ordinate of the data points that represent days of the week
gold_price: y-coordinates of the data points that represent the gold price.
'''
