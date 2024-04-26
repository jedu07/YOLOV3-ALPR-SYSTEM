import os
import matplotlib.pyplot as plt
import numpy as np
def plot_latency_from_file(file_path='latency_analysis.txt'):
    frame_numbers, latency_values = np.loadtxt(file_path, delimiter=',', unpack=True)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, latency_values, marker='o', color='b', label='Latency (seconds)')
    plt.axhline(y=np.mean(latency_values), color='r', linestyle='--', label='Average Latency')

    plt.title('Latency Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # You can also return the average latency if needed
    return np.mean(latency_values)
# Call the function to plot the latency data


plot_latency_from_file()
