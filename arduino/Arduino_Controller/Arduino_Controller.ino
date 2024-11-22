#include <SPI.h>
#include "RF24.h"

const byte address[10] = "ADDRESS01";  // địa chỉ để phát
RF24 radio(7, 8);                      //thay 10 thành 53 với mega
char forward[] = "1";
char backward[] = "2";
char left[] = "3";
char right[] = "4";
char tilt_left[] = "5";
char tilt_right[] = "6";
char stp[] = "0";

void setup() {
  //============================================================Module NRF24

  Serial.begin(9600);
  while (!radio.begin()) {
    Serial.println("Init hardware fail!");
    delay(1000);
  }
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MAX);
  radio.stopListening();

  pinMode(2, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);
}

void loop() {
  if ((digitalRead(2) == LOW) && (digitalRead(3) == LOW)) {
    Serial.println("stp");
    radio.write(&stp, sizeof(stp));
  } else if (digitalRead(3) == LOW) {
    Serial.println("right");
    radio.write(&right, sizeof(right));
  } else if (digitalRead(2) == LOW) {
    Serial.println("left");
    radio.write(&left, sizeof(left));
  }

  char c[5] = "";
  if (Serial.available() > 0) {
    c[0] = Serial.read();
    // Serial.print(c);
  }

  if (c[0] == 35){
    Serial.write(64);
  }

  if (c[0] == 49) {
    // Serial.println("Forward");
    radio.write(&forward, sizeof(forward));
  } else if (c[0] == 50) {
    // Serial.println("Backward");
    radio.write(&backward, sizeof(backward));
  } else if (c[0] == 51) {
    // Serial.println("Left");
    radio.write(&left, sizeof(left));
  } else if (c[0] == 52) {
    // Serial.println("Right");
    radio.write(&right, sizeof(right));
  } else if (c[0] == 53) {
    // Serial.println("Tilt left");
    radio.write(&tilt_left, sizeof(tilt_left));
  } else if (c[0] == 54) {
    // Serial.println("Tilt right");
    radio.write(&tilt_right, sizeof(tilt_right));
  } else if (c[0] == 48) {
    // Serial.println("Stop");
    radio.write(&stp, sizeof(stp));
  }
  delay(150);
}