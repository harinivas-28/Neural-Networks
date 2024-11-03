import numpy as np

a1 = np.array([1, 2, 3])
a2 = np.array([[1],[2],[3]])
print(a1+a2)
'''
Compatibility Rules for Broadcasting
Broadcasting only works with compatible arrays. NumPy compares a set of array dimensions from right to left.

Every set of dimensions must be compatible with the arrays to be broadcastable. A set of dimension lengths is compatible when

one of them has a length of 1 or
they are equal

Broadcastable Shapes
(6, 7) and (6, 7)
(6, 7) and (6, 1)
(6, 7) and (7, )

Non-Broadcastable Shapes
(6, 7) and (7, 6)
(6, 7) and (6, )
'''
print(a2+10, a1+10)
'''
Functions ->	Descriptions
array() ->	creates a matrix
dot() ->	performs matrix multiplication
transpose() ->	transposes a matrix
linalg.inv() ->	calculates the inverse of a matrix
linalg.det() ->	calculates the determinant of a matrix
flatten() ->	transforms a matrix into 1D array
'''
arr2d = np.array([[1, 2, 3],[4, 5 ,6],[7, 8, 9]])
print("Dot Product")
print(np.dot(a1, a2))
print("TRANSPOSE: ")
# print((a1+a2*2+11).transpose()) OR
# print(np.transpose(arr2d)) OR
print(arr2d.T)
# print("INVERSE: ",np.linalg.inv(arr2d)) ERROR
print(arr2d)
mat1 = np.array([[1,2],[3,4]])
print("DET: ", np.round(np.linalg.det(mat1), 3))
print("FLATTEN: ",arr2d.flatten())

# SET OPERATIONs
mat2 = np.array([-1, -2, 0, 3, 4, 5])
print("UNION: ",np.union1d(a1, mat2)) # UNION
print("INTERSECTION: ",np.intersect1d(a1, mat2)) # INTERSECTION
print("DIFFERENCE: ",np.setdiff1d(a1, mat2))
'''
Set Symmetric Difference Operation in NumPy
The symmetric difference between two sets A and B includes all 
elements of A and B without the common elements.
'''
print("SET XOR DIFFERENCE: ",np.setxor1d(a1, mat2))
mat3 = np.array([1, 1, 1,2, 2,3 ,4, 5, 6,3, 2, 1,2, 3, 4,3, 2])
print("UNIQUE: ",np.unique(mat3))

# VECTORIZATION
print(a1+10)
print(a2+arr2d)

def find_sqr(x):
    if x<0: return 0
    else: return x**2

vectorized = np.vectorize(find_sqr)
res = vectorized(mat2)
print(res)
# FANCY INDEXING
select_elements = mat2[[0, 3, -1]] # Represents indexs to select
print(select_elements)
# Sorting using fancy indexing
print(mat3[[np.argsort(mat3)]])
print(mat3[[np.argsort(-mat3)]]) # descending order
# Assigning new values
indices = [0, 3, -1]
new_Vals = [1001, 1004 ,1005]
mat2[indices] = new_Vals
print(mat2)
