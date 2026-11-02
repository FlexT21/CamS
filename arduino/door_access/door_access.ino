// Serial indicator only: no relay or door actuator is connected by this sketch.
// Arduino Nano / Uno: ASCII "0\n" turns pin 13 ON; "1\n" turns it OFF. 9600 baud.
const byte LED_PIN = 13;
char command = '\0';
bool invalidLine = false;
bool carriageReturn = false;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      if (!invalidLine && (command == '1' || command == '0')) {
        digitalWrite(LED_PIN, command == '0' ? HIGH : LOW);
        Serial.print("ACK ");
        Serial.println(command);
      }
      command = '\0';
      invalidLine = false;
      carriageReturn = false;
    } else if (ch == '\r') {
      if (command == '\0' || carriageReturn) invalidLine = true;
      carriageReturn = true;
    } else if (carriageReturn) {
      invalidLine = true;
    } else if (command == '\0' && (ch == '1' || ch == '0')) {
      command = ch;
    } else {
      invalidLine = true;
    }
  }
}
