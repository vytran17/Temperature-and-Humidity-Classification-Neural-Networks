import numpy as np
import pandas as pd
import tensorflow as tf # Convert the model to TensorFlow Lite format
import os 

from everywhereml.code_generators.tensorflow import tf_porter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('temperature_humidity_labels.csv')

# Prepare the input features and target labels
X = data[['Temperature', 'Humidity']].values
y = to_categorical(data['Label'].values)  # One-hot encode the labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(100, input_dim=2, activation='relu'),  # Input layer with 2 inputs and hidden layer with 100 neurons
    Dense(2, activation='softmax')  # Output layer with 2 neurons (for 2 classes)
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Predict the labels for the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate the accuracy score
acc_score = accuracy_score(y_test_classes, y_pred_classes)

print(f"Model Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Model Test Accuracy: {accuracy:.4f}")
print(f"Accuracy Score: {acc_score:.4f}")


# Save the Keras model to HDF5 file
model.save('my_model.h5')

# Create a converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model saved successfully!")

# #CONVERT TO ARDUINO MODEL THE HARD WAY

# # Function to convert to C array
# def convert_to_c_array(tflite_file, header_file):
#     with open(tflite_file, 'rb') as file:
#         tflite_model = file.read()
        
#     with open(header_file, 'w') as file:
#         file.write('const unsigned char model[] = {')
#         file.write(','.join(map(lambda x: hex(x), tflite_model)))
#         file.write('};')
#         file.write(f'\nconst int model_len = {len(tflite_model)};')

# # Use the function to convert the model
# convert_to_c_array('model.tflite', 'model.h')

# print(f"Model converted successfully!")

#CONVERT TO ARDUINO MODEL THE EASY WAY

# Convert using everywhereml's code generator
porter = tf_porter(model, X_train, y_train)

# Generate the C++ code with an instance of the model and a memory arena size of 4096 bytes
cpp_code = porter.to_cpp(instance_name='tempHumidNN', arena_size=4096)

# Save the generated C++ code to a file
with open('TempHumidityModel.h', 'w') as f:
    f.write(cpp_code)

print("C++ code saved successfully!")