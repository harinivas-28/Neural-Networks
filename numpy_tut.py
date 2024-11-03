import numpy as np

arr = np.array([1,2,3,4,5])
print(arr, type(arr), np.__version__)
arr0 = np.array(42)
print(arr0)
arr3D = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3D)
print("Check dimensions of each array")
print(arr0.ndim, arr.ndim, arr3D.ndim)
# higher dimensional arrays
arr5d = np.array([1, 2, 3, 4, 5], ndmin=5)
print(arr5d)
# Accessing array elements
arr2D = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr[3], arr2D[0,1], arr3D[0,1,2], arr2D[1,-1])
'''
The first number represents the first dimension, which contains two arrays:
[[1, 2, 3], [4, 5, 6]]
and:
[[7, 8, 9], [10, 11, 12]]
Since we selected 0, we are left with the first array:
[[1, 2, 3], [4, 5, 6]]

The second number represents the second dimension, which also contains two arrays:
[1, 2, 3]
and:
[4, 5, 6]
Since we selected 1, we are left with the second array:
[4, 5, 6]

The third number represents the third dimension, which contains three values:
4
5
6
Since we selected 2, we end up with the third value:
6
'''
# Array slicing
print(arr[:4])
print(arr[-3:-1])
print(arr[::2]) # with step
print(arr2D[1, :3]) # slicing 2D array
print(arr2D[:,2]) # prints 3rd element in each row
print(arr2D[:,:2]) # prints upto 3rd element from each row

# Data types
print(arr2D.dtype)
str_arr = np.array(['apple', 'orange', 'grapes'])
print(str_arr.dtype)
# create with desired data type
des_arr = np.array([1.24, 2.56, 3.98], dtype='S')
# we can define size as well
des_arr2 = np.array([0, 1, 2, 3, 4], dtype='i4')
print(des_arr, des_arr2)
print(des_arr.dtype, des_arr2.dtype)
'''ValueError: In Python ValueError is raised when the type of
 passed argument to a function is unexpected/incorrect.'''
new_arr = des_arr.astype('f')
print(new_arr)
new_bool_arr = des_arr2.astype(bool)
print(new_bool_arr)
# Array Copy
temp = arr.copy()
arr[0] = 42
'''The copy SHOULD NOT be affected by the changes made to the original array.'''
print(arr, temp)
view = arr.view()
arr[0] = 97
'''The view SHOULD be affected by the changes made to the original array.'''
print(arr, view)
view[3] =25
'''The original array SHOULD be affected by the changes made to the view.'''
print(arr, view)
# Shape of an array
print(arr2D.shape)
'''The example above returns (2, 5), which means that the array has 2 dimensions, where the first dimension has 2 elements and the second has 5.'''
arr_to_shape = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
shapped_arr = arr_to_shape.reshape(4, 3)
print(shapped_arr)
shape_arr3d = arr_to_shape.reshape(2, 3, 2)
print(shape_arr3d)
'''
You are allowed to have one "unknown" dimension.

Meaning that you do not have to specify an exact number for one of the dimensions in the reshape method.

Pass -1 as the value, and NumPy will calculate this number for you.
Note: We can not pass -1 to more than one dimension.
'''
temp_arr = arr_to_shape.reshape(2, -1)
print(temp_arr)
# Flattening arrays
new_flat_arr = arr2D.reshape(-1)
print(new_flat_arr)
# Iterating
# for x in np.nditer(arr2D):
#     print(x)
for x in np.nditer(arr2D[:, ::2]): # printing each element with step
    print(x)
# Enumerated Iteration
for idx, x in np.ndenumerate(arr2D):
    print(idx, x)

# Joining
'''
Joining means putting contents of two or more arrays in a single array.

In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.

We pass a sequence of arrays that we want to join to the concatenate() function, along with the axis. If axis is not explicitly passed, it is taken as 0.
'''
for_join = np.array([10, 9 ,8,7, 6])
join_arr = np.concatenate((arr, arr_to_shape))
print("JOINING: ")
print(join_arr)
join_arr2d = np.concatenate((arr2D, temp_arr), axis=1)
print(join_arr2d)
print("Joining using Stack functions")
join_stack_arr = np.stack((arr, for_join), axis=0)
print(join_stack_arr)
print("Stacking along rows: ")
join_stack_row = np.hstack((arr, for_join))
print(join_stack_row)
print("Stacking along Columns: ")
join_stack_col = np.vstack((arr, for_join))
print(join_stack_col)
print("Stacking along depth(height): ")
join_stack_depth = np.dstack((arr, for_join))
print(join_stack_depth)

# Splitting Arrays
arr_split = np.array_split(arr, 3)
# Note: The return value is a list containing three arrays.
print(arr_split)
# Note: We also have the method split() available but it will not adjust the elements when elements are less in source array for splitting like in example above, array_split() worked properly but split() would fail.
arr_eq_split = np.split(arr_to_shape, 3)
print(arr_eq_split)
# Splitting 2D array
arr2d_split = np.array_split(arr2D, 5)
print(arr2d_split)
# Split the 2-D array into three 2-D arrays.
arr2d_to_split = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr2d = np.array_split(arr2d_to_split, 3) # axis=1(for vertical split)
print(newarr2d)
# hsplit(), vsplit() and dsplit() are used for different types of splitting

# SEARCHING in arrays
arr_to_search = np.array([1, 2, 3, 4, 5, 4, 4])
x1 = np.where(arr_to_search == 4)
x2 = np.where(arr_to_search%2==0)
print(x1)
print(x2)
'''
There is a method called searchsorted() which performs a binary search in the array, and returns the index where the specified value would be inserted to maintain the search order.

The searchsorted() method is assumed to be used on sorted arrays.
'''
x3 = np.searchsorted(arr, 4) # side='right' Use this to search from the right side
print(x3)
x4 = np.searchsorted(arr,[3, 5, 10])
print(x4)
# SORTING Arrays
sort_arr = np.sort(arr_to_search)
# Note: This method returns a copy of the array, leaving the original array unchanged.
print(sort_arr)
print(np.sort(str_arr)) # any kind of datatype
'''
Filtering Arrays
Getting some elements out of an existing array and creating a new array out of them is called filtering.
'''
arr_To_filter = np.array([41, 42, 43, 44])
bool_arr = [True, False, True, False]
filtered_arr = arr_To_filter[bool_arr]
print(filtered_arr)
# Creating filter array
new_arr_to_filter = np.array([1,2,3,4,5,6,4,8,5,0])
filter_arr = []
for m in new_arr_to_filter:
    if m%2==0: filter_arr.append(True)
    else: filter_arr.append(False)

new_filtered = new_arr_to_filter[filter_arr]
print(filter_arr)
print(new_filtered)
# Direct approach
new_filter_arr = new_arr_to_filter%2==1
new_odds = new_arr_to_filter[new_filter_arr]
print(new_filter_arr)
print(new_odds)
