"""
This file is for exploring what a numpy ndarray looks like.
"""
from typing import Any

import numpy as np

"""
------------------------------------------------------------------------------------------------
Introduction to the "ndarray":
- ndarrays are short for n-dimensional arrays
- they can have any number of directions or axes
- the shape of the array specifies the number of dimensions and the "depth" of each dimension
- the dtype is just the type of data stored, such as an int or float
- the buffer specifies what values should be fit into the array
The examples that follow show how these arrays are structured as nested lists of depth n, as well
as how to create these types of nested lists.
-----------------------------------------------------------------------------------------------
"""


def nested_list_ndarray_style(shape: list[int], buffer: list[Any], first=True) -> list[list]:
    """
    Preconditions:
    - len(buffer) >= np.prod(shape)
    >>> nested_list_ndarray_style([3], [1, 2, 3, 4, 5])
    [1, 2, 3]
    >>> nested_list_ndarray_style([2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    [[1, 2], [3, 4]]
    >>> nested_list_ndarray_style([2, 3, 4], [i for i in range(1, 25)])
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
    >>> nested_list_ndarray_style([2, 2, 3, 4], [i for i in range(1, 49)])
    [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], [[[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]], [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]]]
    """
    if len(shape) == 1:
        return buffer[:shape[0]]
    else:
        nested_list = []
        new_shape = shape[1:]
        nested_list_length = shape[0]
        buffer_elements_added = 0
        for i in range(nested_list_length):
            nested_list.append(nested_list_ndarray_style(new_shape, buffer[buffer_elements_added:], False))
            buffer_elements_added += np.prod(new_shape)
        return nested_list


# create some one-dimensional arrays
print(np.ndarray(shape=(3,), dtype=int, buffer=np.array([1, 2, 3])))
print(np.ndarray(shape=(3,), dtype=int, buffer=np.array([2, 4, 6, 8])))
print(np.ndarray(shape=(10,), dtype=int, buffer=np.array([i ** 2 for i in range(10)])))

# create a two-dimensional array
print(np.ndarray(shape=(3, 3), dtype=int, buffer=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])))
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# create a three-dimensional array
print(np.ndarray(shape=(1, 2, 3), dtype=int, buffer=np.array([1, 2, 3, 4, 5, 6])))
# [[[1 2 3]
#   [4 5 6]]]

# create another three-dimensional array
print(np.ndarray(shape=(2, 3, 4), dtype=int, buffer=np.array([i for i in range(1, 25)])))
# [[[ 1  2  3  4]
#   [ 5  6  7  8]
#   [ 9 10 11 12]]
#  [[13 14 15 16]
#   [17 18 19 20]
#   [21 22 23 24]]]

# create a four-dimensional array
print(np.ndarray(shape=(2, 3, 4, 5), dtype=int, buffer=np.array([i for i in range(2 * 3 * 4 * 5)])))

# [[[[  0   1   2   3   4]
#    [  5   6   7   8   9]
#    [ 10  11  12  13  14]
#    [ 15  16  17  18  19]]
#   [[ 20  21  22  23  24]
#    [ 25  26  27  28  29]
#    [ 30  31  32  33  34]
#    [ 35  36  37  38  39]]
#   [[ 40  41  42  43  44]
#    [ 45  46  47  48  49]
#    [ 50  51  52  53  54]
#    [ 55  56  57  58  59]]]
#  [[[ 60  61  62  63  64]
#    [ 65  66  67  68  69]
#    [ 70  71  72  73  74]
#    [ 75  76  77  78  79]]
#   [[ 80  81  82  83  84]
#    [ 85  86  87  88  89]
#    [ 90  91  92  93  94]
#    [ 95  96  97  98  99]]
#   [[100 101 102 103 104]
#    [105 106 107 108 109]
#    [110 111 112 113 114]
#    [115 116 117 118 119]]]]
# %%
