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

byte LecturaUID[7];
bool camaraVerificada = false;
unsigned long momentoVerificacionCamara = 0;

// Tarjetas autorizadas
byte Usuario1[4] = {0x04, 0x0F, 0x4B, 0xAC};
byte Usuario2[4] = {0x08, 0x2F, 0xC7, 0xEC};

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

  if (signal == "ACCEPTED") {
    camaraVerificada = true;
    momentoVerificacionCamara = millis();
    digitalWrite(LEDR, LOW);
    Serial.println("Camara verificada; esperando tarjeta RFID...");
  } else if (signal == "DENIED") {
    camaraVerificada = false;
    Serial.println("Camara no verificada; esperando una nueva persona...");
  }
}

void verificarExpiracionCamara() {
  if (
    camaraVerificada &&
    millis() - momentoVerificacionCamara >= CONFIRMACION_CAMARA_MS
  ) {
    camaraVerificada = false;
    Serial.println("Verificacion de camara expirada");
  }
}

void revisarTarjeta() {
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return;
  }

  if (!mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  Serial.print("UID:");
  for (byte i = 0; i < mfrc522.uid.size && i < sizeof(LecturaUID); i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) {
      Serial.print(" 0");
    } else {
      Serial.print(" ");
    }

    Serial.print(mfrc522.uid.uidByte[i], HEX);
    LecturaUID[i] = mfrc522.uid.uidByte[i];
  }
  Serial.println();

  bool tarjetaAutorizada =
    comparaUID(LecturaUID, mfrc522.uid.size, Usuario1, 4) ||
    comparaUID(LecturaUID, mfrc522.uid.size, Usuario2, 4);

  if (!camaraVerificada) {
    accesoDenegado("RFID sin camara verificada");
  } else if (tarjetaAutorizada) {
    accesoPermitido("RFID");
  } else {
    accesoDenegado("RFID");
  }

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  delay(300);
}

void accesoPermitido(const char* origen) {
  camaraVerificada = false;

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
  camaraVerificada = false;

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

boolean comparaUID(
  byte lectura[],
  byte lecturaSize,
  byte usuario[],
  byte usuarioSize
) {
  if (lecturaSize != usuarioSize) {
    return false;
  }

  for (byte i = 0; i < lecturaSize; i++) {
    if (lectura[i] != usuario[i]) {
      return false;
    }
  }

  return true;
}
