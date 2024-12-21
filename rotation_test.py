import numpy as np
from scipy import interpolate

# Define datetime64 array for x
x = np.array(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'], dtype='datetime64[D]')
y = np.array([10, np.nan, 30, 40])

# Mask for missing values
mask = np.isnan(y)

# Interpolate
y[mask] = interpolate.interp1d(x[~mask].astype('datetime64[D]').astype('int'), y[~mask], kind='linear')(
    x[mask].astype('datetime64[D]').astype('int'))

print(y)
