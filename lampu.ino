#include <TM1637Display.h>
#include <SoftwareSerial.h>
#include <DFRobotDFPlayerMini.h>

// =====================
// TM1637
// =====================
#define CLK 2
#define DIO 3
TM1637Display display(CLK, DIO);

// =====================
// LAMPU & TOMBOL
// =====================
#define RED     4
#define GREEN   5
#define YELLOW  7
#define BUTTON_PIN 8

// =====================
// MP3
// =====================
SoftwareSerial mp3Serial(10, 11); // RX, TX Arduino
DFRobotDFPlayerMini mp3;

// =====================
// VARIABLE
// =====================
unsigned long lastTick = 0;
int countdown = 0;
bool isCounting = false;
String serialBuffer = "";

int buttonState = HIGH;
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

enum LampState { STATE_RED, STATE_GREEN, STATE_YELLOW };
LampState currentState = STATE_RED;

// =====================
// FUNGSI
// =====================
void setMerah() {
  digitalWrite(RED, HIGH);
  digitalWrite(YELLOW, LOW);
  digitalWrite(GREEN, LOW);
  display.clear();
  isCounting = false;
  currentState = STATE_RED;
  Serial.println("ACK_MERAH");
}

void setHijau(int durasi) {
  digitalWrite(RED, LOW);
  digitalWrite(YELLOW, LOW);
  digitalWrite(GREEN, HIGH);
  mp3.play(1); // /mp3/0001.mp3
  countdown = durasi;
  display.showNumberDec(countdown, true);
  isCounting = true;
  lastTick = millis();
  currentState = STATE_GREEN;
  Serial.println("ACK_HIJAU");
}

void setKuning(int durasi) {
  digitalWrite(RED, LOW);
  digitalWrite(YELLOW, HIGH);
  digitalWrite(GREEN, LOW);
  countdown = durasi;
  display.showNumberDec(countdown, true);
  isCounting = true;
  lastTick = millis();
  currentState = STATE_YELLOW;
  Serial.println("ACK_KUNING");
}

void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd == "MERAH") {
    setMerah();
  } else if (cmd.startsWith("HIJAU:")) {
    int dur = cmd.substring(6).toInt();
    if (dur <= 0) dur = 15;
    setHijau(dur);
  } else if (cmd.startsWith("KUNING:")) {
    int dur = cmd.substring(7).toInt();
    if (dur <= 0) dur = 3;
    setKuning(dur);
  } else if (cmd == "HIJAU") {
    setHijau(15); // fallback
  } else if (cmd == "KUNING") {
    setKuning(3); // fallback
  }
}

// =====================
// SETUP
// =====================
void setup() {
  Serial.begin(9600);
  
  pinMode(RED, OUTPUT);
  pinMode(YELLOW, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  display.setBrightness(7);

  // MP3 INIT
  mp3Serial.begin(9600);
  if (!mp3.begin(mp3Serial)) {
    Serial.println("ERROR_MP3");
  } else {
    mp3.volume(20);
  }

  setMerah();
}

// =====================
// LOOP
// =====================
void loop() {
  // 1. NON-BLOCKING SERIAL READ
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        processCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }

  // 2. BACA TOMBOL MANUAL (DEBOUNCE)
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      // Trigger Python HANYA saat kondisi merah (supaya tidak double trigger)
      if (buttonState == LOW && currentState == STATE_RED) {
        Serial.println("BUTTON_PRESSED");
      }
    }
  }
  lastButtonState = reading;

  // 3. COUNTDOWN DISPLAY
  if (isCounting && millis() - lastTick >= 1000) {
    lastTick = millis();
    countdown--;

    if (countdown >= 0) {
      display.showNumberDec(countdown, true);
    } else {
      // Waktu habis
      isCounting = false;
      display.clear();
      // Fallback: Jika setelah countdown habis tidak ada instruksi serial,
      // Arduino otomatis kembali ke merah demi keselamatan lalu lintas.
      if (currentState != STATE_RED) {
        setMerah();
      }
    }
  }
}
