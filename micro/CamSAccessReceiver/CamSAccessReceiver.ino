#include <SPI.h>
#include <MFRC522.h>

#define RST_PIN 9
#define SS_PIN 10

MFRC522 mfrc522(SS_PIN, RST_PIN);

// Pines
#define LEDG 8
#define BUZZER 4
#define LEDR 6
#define RELE 7

// Configuracion del rele
#define RELE_ON HIGH
#define RELE_OFF LOW
#define CONFIRMACION_CAMARA_MS 10000UL

String tarjetaEsperada;
String solicitudRegistro;
unsigned long momentoSenal = 0;

void setup() {
  Serial.begin(9600);

  pinMode(RELE, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  digitalWrite(RELE, RELE_OFF);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDR, LOW);
  digitalWrite(BUZZER, LOW);

  SPI.begin();
  mfrc522.PCD_Init();

  Serial.println("Sistema RFID y serial listo");
  Serial.println("Acerque una tarjeta...");
}

void loop() {
  recibirSenalSerial();
  verificarExpiracionCamara();
  revisarTarjeta();
}

void recibirSenalSerial() {
  if (!Serial.available()) {
    return;
  }

  String signal = Serial.readStringUntil('\n');
  signal.trim();

  tarjetaEsperada = "";
  solicitudRegistro = "";
  digitalWrite(LEDR, LOW);
  momentoSenal = millis();
  if (signal.startsWith("ENROLL:") && signal.length() == 19) {
    solicitudRegistro = signal.substring(7);
    digitalWrite(LEDR, HIGH);
  } else if (signal.startsWith("VERIFY:")) {
    tarjetaEsperada = signal.substring(7);
  }
}

void verificarExpiracionCamara() {
  unsigned long elapsed = millis() - momentoSenal;
  if (elapsed >= CONFIRMACION_CAMARA_MS) tarjetaEsperada = "";
  if (solicitudRegistro.length() && elapsed >= 20000UL) {
    solicitudRegistro = "";
    digitalWrite(LEDR, LOW);
  }
}

void revisarTarjeta() {
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return;
  }

  if (!mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  String uid;
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    byte valor = mfrc522.uid.uidByte[i];
    if (valor < 0x10) uid += '0';
    uid += String(valor, HEX);
  }
  uid.toUpperCase();

  if (solicitudRegistro.length()) {
    Serial.print("CARD:");
    Serial.print(solicitudRegistro);
    Serial.print(':');
    Serial.println(uid);
    solicitudRegistro = "";
    digitalWrite(LEDR, LOW);
  } else if (uid.length() && uid == tarjetaEsperada) {
    accesoPermitido("RFID");
  } else {
    accesoDenegado("RFID");
  }

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  delay(300);
}

void accesoPermitido(const char* origen) {
  tarjetaEsperada = "";

  Serial.print("ACCESO PERMITIDO (");
  Serial.print(origen);
  Serial.println(")");

  digitalWrite(RELE, RELE_ON);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, LOW);
  digitalWrite(BUZZER, HIGH);
  delay(5000);

  digitalWrite(RELE, RELE_OFF);
  digitalWrite(LEDG, LOW);
  digitalWrite(BUZZER, LOW);
  Serial.println("Cerradura cerrada");
}

void accesoDenegado(const char* origen) {
  tarjetaEsperada = "";

  Serial.print("ACCESO DENEGADO (");
  Serial.print(origen);
  Serial.println(")");

  digitalWrite(RELE, RELE_OFF);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDR, HIGH);

  for (byte pitido = 0; pitido < 2; pitido++) {
    digitalWrite(BUZZER, HIGH);
    delay(200);
    digitalWrite(BUZZER, LOW);
    if (pitido == 0) {
      delay(200);
    }
  }

  delay(700);
  digitalWrite(LEDR, LOW);
}
