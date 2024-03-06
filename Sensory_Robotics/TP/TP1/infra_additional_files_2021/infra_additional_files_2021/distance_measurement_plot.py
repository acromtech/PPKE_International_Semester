import numpy as np
import matplotlib.pyplot as plt

# depending of your Python version and numpy version, the last two options of np.load suspectedly necessary:
[measurement_array, max_distance] = np.load('./numpy_array_of_infra_measurement_alu.npy', allow_pickle='False', encoding='bytes')
print(measurement_array.shape)
print(max_distance)

# please plot here the loaded array




