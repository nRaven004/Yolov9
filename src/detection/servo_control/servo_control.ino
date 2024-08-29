#include <Servo.h>

Servo myServo;  // Create a Servo object

void setup() {
  myServo.attach(0); 

}

void loop() {
  myServo.write(150);
}

//keep base:20 degree, middle:170 , hand 80
//grap base: 80, mid: 
