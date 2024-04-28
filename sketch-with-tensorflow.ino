#include <DHT.h>
#include <LiquidCrystal.h>
#include "TempHumidityModel.h"

// Constants
#define DHTPIN 2       // What digital pin the DHT22 is connected to
#define DHTTYPE DHT22  // There are multiple types of DHT sensors

// Initialize the DHT sensor
DHT dht(DHTPIN, DHTTYPE);

// Initialize the LCD library with the numbers of the interface pins
LiquidCrystal lcd(12, 11, 10, 9, 8, 7);

void setup() {
  Serial.begin(9600); // Start the serial communication
  while (!tempHumidNN.begin()) {
        Serial.print("Error in NN initialization: ");
        Serial.println(tempHumidNN.getErrorMessage());
  }

  // Set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  // Print a message to the LCD.
  lcd.print("Temp:       C");
  lcd.setCursor(0, 1);
  lcd.print("Humid:       %");

  // Start the DHT sensor
  dht.begin();
}

void loop() {
  // Wait a few seconds between measurements.
  delay(2000);

  // Reading temperature or humidity takes about 250 milliseconds!
  float humidity = dht.readHumidity();
  // Read temperature as Celsius (the default)
  float temperature = dht.readTemperature();

  // Check if any reads failed and exit early (to try again).
  if (isnan(humidity) || isnan(temperature)) {
    lcd.setCursor(0, 0);
    lcd.print("Error read sensor");
    return;
  }

  // Print the temperature to the LCD
  lcd.setCursor(6, 0); // Set the cursor to column 6, row 0
  lcd.print(temperature);
  // Print the humidity to the LCD
  lcd.setCursor(7, 1); // Set the cursor to column 7, row 1
  lcd.print(humidity);


  // Prepare the inputs for prediction
  float inputs[2] = {temperature, humidity};
  tempHumidNN.predict(inputs);

  // Get the prediction result
  int prediction = tempHumidNN.getPrediction();

  // Display the prediction on the LCD
  lcd.setCursor(0, 1); // Set the cursor to the beginning of the second row
  if (prediction == 0) {
    lcd.print("Normal       "); // Extra spaces to clear out any previous text
  } else if (prediction == 1) {
    lcd.print("Extreme      "); // Extra spaces to clear out any previous text
  }
}

