// Arduino Uno: USB Serial at 9600 baud. ASCII '0' turns pin 13 ON.
const byte LED_PIN = 13;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    const int command = Serial.read();
    if (command == '0') {
      digitalWrite(LED_PIN, HIGH);
      Serial.println("LED_ON");
    } else if (command == '1') {
      digitalWrite(LED_PIN, LOW);
      Serial.println("LED_OFF");
    }
  }
}
