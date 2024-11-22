#include <SPI.h>
#include "RF24.h"

#define IN1 A0
#define IN2 A1
#define IN3 A2
#define IN4 A3

#define CE_PIN 7
#define CSN_PIN 8

RF24 radio(CE_PIN, CSN_PIN);
const byte address[10] = "ADDRESS01";  // địa chỉ phát
byte command;
bool isForward = false;
int speedMax = 255;
int FBMaxSPD = 130;
int offset = 0;

void forward(int spd1, int spd3) {

  analogWrite(IN1, spd1);
  analogWrite(IN2, 0);
  analogWrite(IN3, spd3);
  analogWrite(IN4, 0);
  // delay(120);
}

void backward(int spd2, int spd4) {
  analogWrite(IN1, 0);
  analogWrite(IN2, spd2);
  analogWrite(IN3, 0);
  analogWrite(IN4, spd4);
  // delay(120);
}

void tilt_left(int spd) {
  analogWrite(IN1, 100);
  analogWrite(IN2, 0);
  analogWrite(IN3, spd);
  analogWrite(IN4, 0);
  // delay(120);
}

void tilt_right(int spd) {
  analogWrite(IN1, spd);
  analogWrite(IN2, 0);
  analogWrite(IN3, 100);
  analogWrite(IN4, 0);
  // delay(120);
}

void left(int spd) {
  analogWrite(IN1, 0);
  analogWrite(IN2, spd);
  analogWrite(IN3, spd);
  analogWrite(IN4, 0);
  // delay(120);
}

void right(int spd) {
  analogWrite(IN1, spd);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, spd);
  // delay(120);
}
void stp() {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 0);
  // delay(120);
}

void setup() {

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(9600);

  Serial.println("LoRa Receiver");
  Serial.println("Wait for LORA init");

  while (!radio.begin()) {
    Serial.println(F("radio hardware is not responding!!"));
    delay(1000);
  }
  radio.openReadingPipe(0, address);
  radio.setPALevel(RF24_PA_MAX);
  radio.startListening();
}
char txt[5];
void loop() {
  // try to parse packet
  int i = 0;

  if (radio.available()) {  // is there a payload? get the pipe number that recieved it

    radio.read(&txt, sizeof(txt));
    Serial.println(txt[0]);
  }

  if (txt[0] == 49) {
    Serial.println("Go forward");
    isForward = true;

    forward(speedMax, speedMax);
    delay(4);
    forward(0, speedMax);
    delay(2);
    // delayMicroseconds(750);
    forward(0, 0);
    delay(3);

  } else if (txt[0] == 50) {
    Serial.println("Go backward");
    // for (i = i; i < speedMax; i = i + 25) {
    //   backward(i);
    //   delay(5);
    //   backward(0);
    //   delay(5);
    // }
    // for (int j=0; j<)
    // backward(speedMax);
    // delay(100);
    // if (isForward) txt[0] = 49;
    // else txt[0] = 48;

    backward(speedMax, speedMax);
    delay(4);
    backward(speedMax, 0);
    delay(2);
    // delayMicroseconds(900);
    backward(0, 0);
    delay(3);

  } else if (txt[0] == 51) {
    Serial.println("Turn left");

    if (isForward) {
      i = speedMax;
    }
    for (i = i; i < speedMax * 2; i = i + 25) {
      if (i >= speedMax) left(speedMax);
      else left(i);
      delay(5);
      left(0);
      delay(5);
    }
    if (isForward) txt[0] = 49;
    else txt[0] = 48;

  } else if (txt[0] == 52) {
    Serial.println("Turn right");
    if (isForward) {
      i = speedMax;
    }
    for (i = i; i < speedMax * 2; i = i + 25) {
      if (i >= speedMax) right(speedMax);
      else right(i);
      delay(5);
      right(0);
      delay(5);
    }
    if (isForward) txt[0] = 49;
    else txt[0] = 48;

  } else if (txt[0] == 53) {
    Serial.println("Tilt left");
    if (isForward) {
      i = speedMax;
    }
    for (i = i; i < speedMax * 3; i = i + 25) {
      if (i >= speedMax) tilt_left(speedMax);
      else tilt_left(i);
      delay(5);
      tilt_left(0);
      delay(3);
    }
    if (isForward) txt[0] = 49;
    else txt[0] = 48;

  } else if (txt[0] == 54) {
    Serial.println("Tilt right");
    if (isForward) {
      i = speedMax;
    }
    for (i = i; i < speedMax * 3; i = i + 25) {
      if (i >= speedMax) tilt_right(speedMax);
      else tilt_right(i);
      delay(5);
      tilt_right(0);
      delay(3);
    }
    if (isForward) txt[0] = 49;
    else txt[0] = 48;

  } else {
    stp();
    isForward = false;
  }
}