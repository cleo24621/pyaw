from pywt import cwt

# Create a signal
import numpy as np
import matplotlib.pyplot as plt
signal = np.cumsum(np.random.randn(1000))
