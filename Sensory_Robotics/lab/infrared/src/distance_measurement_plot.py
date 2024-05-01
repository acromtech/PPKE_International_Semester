import numpy as np
import matplotlib.pyplot as plt

def plot_mesurement(text,i):
    [measurement_array, max_distance] = np.load(f'./numpy_array_of_infra_measurement_{text}.npy', allow_pickle=True, encoding='bytes')

    measurement_array = measurement_array[::-1]
    phototransistor_values = measurement_array[:, 0]
    photodiode_values = measurement_array[:, 1]

    plt.subplot(1,2,i)
    plt.plot(photodiode_values, label='Photodiode', color='blue')
    plt.plot(phototransistor_values, label='Phototransistor', color='green')
    plt.title(f'Infra Measurement for {text}')
    plt.xlabel('Distance (cm)')
    plt.ylabel('ADC Values')
    plt.xlim(0, max_distance*1000)
    plt.ylim(0, 255)
    plt.legend()

plt.figure(figsize=(16, 6))
plot_mesurement('paper',1)
plot_mesurement('alu',2)
plt.show()
