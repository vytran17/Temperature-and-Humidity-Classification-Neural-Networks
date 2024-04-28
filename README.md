# Temperature and Humidity Classification with Neural Networks

## Overview

This repository hosts the Python and Arduino code for a project that involves creating a machine learning model to classify temperature and humidity conditions as "Normal" or "Extreme." The project uses Keras and TensorFlow for modeling and TFLite for deployment on Arduino using Wokwi simulation.

## Project Structure

- `Python Codes`: Contains the Python script to generate and label the dataset, develop and train the neural network model, and evaluate its performance.
- `Keras TensorFlow Machine Learning`: Details the creation of a neural network model with TensorFlow and its training process.
- `Conversion and Deploy`: Discusses the conversion of the TensorFlow model to TensorFlow Lite format, further conversion to a `.h` file, and deployment on Wokwi for Arduino-based real-time prediction.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-Learn

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/vytran17/Temperature-and-Humidity-Classification-Neural-Networks
2. Install required Python packages:
   ```sh
   pip install tensorflow pandas numpy scikit-learn

### Generating the Dataset
Run the generate_dataset.py script to create a dataset of temperature and humidity samples with assigned labels: 
python generate_dataset.py

### Training the Model
Execute the train_model.py script to train the neural network model and evaluate its performance:
python train_model.py

###  Model Conversion and Deployment
Follow the instructions in the convert_deploy.py script to convert the trained model to TFLite and Arduino format and deploy it for real-time prediction on Wokwi:
python convert_deploy.py

###  Usage
Once deployed on Wokwi, the Arduino setup reads real-time temperature and humidity data and uses the model to predict the condition. Depending on the prediction:

- 0 (Normal): Displays "Normal" on the LCD.
- 1 (Extreme): Displays "Extreme" on the LCD.

###  License
This project is licensed under the MIT License - see the LICENSE file for details.

###  Acknowledgments
- Wokwi, for providing a platform to simulate Arduino projects online.




