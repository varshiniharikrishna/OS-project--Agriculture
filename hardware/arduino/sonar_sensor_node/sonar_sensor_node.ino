/*
 * AgriEdge Intelligence OS — Ultrasonic Sonar Hardware Node
 * ═══════════════════════════════════════════════════════════════
 * Board: Arduino Uno
 * Sensor: 4-Wire Ultrasonic Sonar Distance Sensor (HC-SR04)
 *
 * 4-WIRE PIN CONNECTIONS:
 *   1. VCC   ──>  Arduino 5V
 *   2. GND   ──>  Arduino GND
 *   3. TRIG  ──>  Arduino Digital Pin 9 (Output trigger pulse)
 *   4. ECHO  ──>  Arduino Digital Pin 10 (Input echo pulse)
 *
 * VISUAL INDICATOR:
 *   • Built-in LED on Pin 13 lights UP whenever obstacle/crop is detected (< 15 cm)
 *
 * SERIAL PROTOCOL:
 *   • Baud Rate: 9600
 *   • Emits standard JSON packet with live distance in cm
 *   • Threshold: < 15.0 cm triggers OBSTACLE_DETECTED and queues Leaf_Inference task in OS
 */

const int TRIG_PIN = 9;         // Sonar TRIG connected to Digital Pin 9
const int ECHO_PIN = 10;        // Sonar ECHO connected to Digital Pin 10
const int LED_PIN = 13;         // Built-in LED on Arduino Uno

// Detection threshold: object closer than 15 cm is treated as crop leaf in proximity
const float PROXIMITY_THRESHOLD_CM = 15.0;

bool lastObstacleState = false;
unsigned long lastMeasureTime = 0;
const unsigned long measureInterval = 200; // Measure distance every 200ms

unsigned long lastHeartbeatTime = 0;
const unsigned long heartbeatInterval = 2000; // 2-second heartbeat
unsigned long eventCounter = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port on USB boards
  }

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);

  // Power-on indicator (3 quick blinks)
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }

  // Send initial startup JSON
  Serial.println("{\"device\":\"ARDUINO_UNO\",\"sensor\":\"SONAR_HC_SR04\",\"status\":\"ONLINE\",\"threshold_cm\":15.0}");
}

float measureDistanceCM() {
  // Clear trigger
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Send 10 microsecond HIGH pulse
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo pulse duration (timeout 30ms = ~5 meters)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration <= 0) {
    return 999.0; // Out of range or no echo received
  }

  // Speed of sound = 343 m/s = 0.0343 cm/microsecond
  // Distance = (duration * 0.0343) / 2
  float distance = (duration * 0.0343) / 2.0;
  return distance;
}

void loop() {
  // 1. Check for incoming serial commands from Python OS
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "PING" || cmd == "STATUS") {
      float d = measureDistanceCM();
      bool isObstacle = (d > 0.0 && d <= PROXIMITY_THRESHOLD_CM);
      Serial.print("{\"type\":\"PONG\",\"device\":\"ARDUINO_UNO\",\"sensor\":\"SONAR_HC_SR04\",\"distance_cm\":");
      Serial.print(d, 1);
      Serial.print(",\"obstacle\":");
      Serial.print(isObstacle ? "true" : "false");
      Serial.print(",\"raw\":");
      Serial.print(isObstacle ? 0 : 1);
      Serial.println("}");
    }
  }

  // 2. Periodic Sonar Distance Measurement
  if (millis() - lastMeasureTime >= measureInterval) {
    lastMeasureTime = millis();
    float distance = measureDistanceCM();
    bool obstacleNow = (distance > 0.0 && distance <= PROXIMITY_THRESHOLD_CM);

    // If state changed (edge detection)
    if (obstacleNow != lastObstacleState) {
      lastObstacleState = obstacleNow;
      eventCounter++;

      if (obstacleNow) {
        digitalWrite(LED_PIN, HIGH); // Turn ON LED
        sendSonarPacket(distance, true, "OBSTACLE_DETECTED");
      } else {
        digitalWrite(LED_PIN, LOW);  // Turn OFF LED
        sendSonarPacket(distance, false, "PATH_CLEAR");
      }
    }
  }

  // 3. Periodic Heartbeat every 2 seconds
  if (millis() - lastHeartbeatTime >= heartbeatInterval) {
    lastHeartbeatTime = millis();
    float currentDist = measureDistanceCM();
    bool currentObstacle = (currentDist > 0.0 && currentDist <= PROXIMITY_THRESHOLD_CM);
    sendSonarPacket(currentDist, currentObstacle, "HEARTBEAT");
  }
}

void sendSonarPacket(float dist, bool isObstacle, const char* eventType) {
  Serial.print("{\"device\":\"ARDUINO_UNO\",\"sensor\":\"SONAR_HC_SR04\",\"distance_cm\":");
  Serial.print(dist, 1);
  Serial.print(",\"obstacle\":");
  Serial.print(isObstacle ? "true" : "false");
  Serial.print(",\"raw\":");
  Serial.print(isObstacle ? 0 : 1);
  Serial.print(",\"state\":\"");
  Serial.print(isObstacle ? "DETECTED" : "CLEAR");
  Serial.print("\",\"event\":\"");
  Serial.print(eventType);
  Serial.print("\",\"count\":");
  Serial.print(eventCounter);
  Serial.println("}");
}
