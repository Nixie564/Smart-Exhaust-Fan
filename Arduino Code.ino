#define PIN_MQ135 13 // MQ135 Analog Input Pin
#define DHTPIN 12 // DHT Digital Input Pin
#define DHTTYPE DHT22 // DHT11 or DHT22, depends on your sensor

#include <MQ135.h>
#include <DHT.h>

MQ135 mq135_sensor(PIN_MQ135);
DHT dht(DHTPIN, DHTTYPE);

float temperature, humidity; // Temp and Humid floats, will be measured by the DHT

void setup() {
  Serial.begin(115200);
  dht.begin();
  delay(8000);
}

void loop() {
  delay(2000);
  humidity = dht.readHumidity();
  humidity = 0.8182 * humidity + 14.575;
  temperature = dht.readTemperature();
  temperature = 0.7727 * temperature + 5.6364;

  // Check if any reads failed and exit early (to try again).
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }
  
  mq135_sensor.getCorrectionFactor(temperature, humidity);
  mq135_sensor.getRZero();
  mq135_sensor.getCorrectedRZero(temperature, humidity);
  mq135_sensor.getResistance();
  mq135_sensor.getCorrectedResistance(temperature, humidity);
  mq135_sensor.getPPM();
  int correctedPPM = mq135_sensor.getCorrectedPPM(temperature, humidity);
  Serial.println(correctedPPM);
}
