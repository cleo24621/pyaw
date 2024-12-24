# encoding = UTF-8
"""
@USER: cleo
@DATE: 2024/12/24
@DESCRIPTION: 
"""

import numpy as np

# Given ndarray a
a = np.linspace(-180, 180, 50)

# Create ndarray b using slicing and vectorized averaging
b = (a[:-1] + a[1:]) / 2

# Print the resulting arrays
print("ndarray a:\n", a)
print("\nndarray b:\n", b)
print("\nShape of a:", a.shape)
print("\nShape of b:", b.shape)