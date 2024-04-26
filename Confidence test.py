import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file
file_path = 'C:\\Users\\jed\\PycharmProjects\\pythonProject\\yolov4-custom-functions-master\\YOLOV3-ALPR-SYSTEM\\results\\results_20240425-063602.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Remove rows with NaN values
df_clean = df.dropna(subset=['Confidence'])

# Sort the DataFrame by confidence scores
df_clean.sort_values(by='Confidence')

# Calculate frequency for each confidence score
frequency = df_clean['Confidence'].value_counts().sort_index()

# Calculate average confidence score
average_confidence = df_clean['Confidence'].mean()

# Plot line graph
plt.figure(figsize=(10, 6))
plt.plot(frequency.index, frequency.values, marker='o', linestyle='-', label='Frequency')
plt.axvline(x=average_confidence, color='red', linestyle='--', label=f'Average Confidence: {average_confidence:.2f}')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution (Line Graph)')
plt.legend()
plt.grid(True)
plt.show()
