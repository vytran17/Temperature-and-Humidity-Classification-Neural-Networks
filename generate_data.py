import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Generate 1000 random samples for temperature and humidity
temperature_samples = np.random.uniform(-40, 80, 1000)  # Range from -40 to 80
humidity_samples = np.random.uniform(0, 100, 1000)      # Range from 0 to 100

# Randomly assign binary labels (0 or 1)
labels = np.random.randint(2, size=1000)  # 0 or 1

# Create a DataFrame to hold the dataset
data = pd.DataFrame({
    'Temperature': temperature_samples,
    'Humidity': humidity_samples,
    'Label': labels
})

# Print the first few rows of the dataset
print(data.head())

# Save the dataset to a CSV file
data.to_csv('temperature_humidity_labels.csv', index=False)
