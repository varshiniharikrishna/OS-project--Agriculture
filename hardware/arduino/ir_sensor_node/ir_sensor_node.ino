/*
 * AgriEdge Intelligence OS — Arduino Hardware Node
 * ═══════════════════════════════════════════════════════════════
 * Board: Arduino Uno
 * Sensor: Active-Low Infrared (IR) Obstacle Avoidance Sensor
 *
 * Physical Wiring:
 *   • Sensor VCC  ──>  Arduino 5V
 *   • Sensor GND  ──>  Arduino GND
 *   • Sensor OUT  ──>  Arduino Digital Pin 2 (Interrupt/Input)
 *
 * Visual Indicator:
 *   • On-board LED (Pin 13) lights up when obstacle/crop is detected.
 *
 * Serial Protocol:
 *   • Baud Rate: 9600
 *   • Format: JSON strings terminated with newline (\n)
 *   • Events: Instant trigger on edge transition + 2000ms heartbeat
 *   • Commands accepted from Host: "PING\n" -> responds with "PONG"
 */

const int IR_PIN = 2;          // OUT pin connected to Digital Pin 2
const int LED_PIN = 13;        // Built-in LED on Arduino Uno

int lastSensorState = -1;      // Previous debounced state (HIGH=Clear, LOW=Detected)
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // 50ms debounce window

unsigned long lastHeartbeatTime = 0;
const unsigned long heartbeatInterval = 2000; // 2-second heartbeat
unsigned long eventCounter = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect (native USB boards)
  }

  pinMode(IR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);

  // Initial power-up blink indicator (3 quick blinks)
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }

  // Read initial sensor state
  lastSensorState = digitalRead(IR_PIN);
  sendSensorState(lastSensorState, "INIT");
}

void loop() {
  // 1. Check for incoming serial commands from Host Python backend
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "PING" || cmd == "STATUS") {
      int currentState = digitalRead(IR_PIN);
      bool isObstacle = (currentState == LOW);
      Serial.print("{\"type\":\"PONG\",\"device\":\"ARDUINO_UNO\",\"sensor\":\"IR_OBSTACLE\",\"pin\":2,\"obstacle\":");
      Serial.print(isObstacle ? "true" : "false");
      Serial.print(",\"raw\":");
      Serial.print(currentState);
      Serial.println("}");
    }
  }

  // 2. Read physical IR sensor with debouncing
  int reading = digitalRead(IR_PIN);

  if (reading != lastSensorState) {
    if ((millis() - lastDebounceTime) > debounceDelay) {
      lastSensorState = reading;
      lastDebounceTime = millis();
      eventCounter++;

      // Active-low: LOW indicates obstacle/crop detected by reflected IR beam
      if (reading == LOW) {
        digitalWrite(LED_PIN, HIGH); // LED ON
        sendSensorState(reading, "OBSTACLE_DETECTED");
      } else {
        digitalWrite(LED_PIN, LOW);  // LED OFF
        sendSensorState(reading, "PATH_CLEAR");
      }
    }
  }

  // 3. Periodic Heartbeat to maintain host OS connection health
  if (millis() - lastHeartbeatTime > heartbeatInterval) {
    lastHeartbeatTime = millis();
    sendHeartbeat(lastSensorState);
  }
}

void sendSensorState(int rawValue, const char* eventType) {
  bool obstacle = (rawValue == LOW); // Active-low logic
  Serial.print("{\"device\":\"ARDUINO_UNO\",\"sensor\":\"IR_OBSTACLE\",\"pin\":2,\"event\":\"");
  Serial.print(eventType);
  Serial.print("\",\"state\":\"");
  Serial.print(obstacle ? "DETECTED" : "CLEAR");
  Serial.print("\",\"obstacle\":");
  Serial.print(obstacle ? "true" : "false");
  Serial.print(",\"raw\":");
  Serial.print(rawValue);
  Serial.print(",\"seq\":");
  Serial.print(eventCounter);
  Serial.print(",\"uptime_ms\":");
  Serial.print(millis());
  Serial.println("}");
}

void sendHeartbeat(int rawValue) {
  bool obstacle = (rawValue == LOW);
  Serial.print("{\"device\":\"ARDUINO_UNO\",\"sensor\":\"IR_OBSTACLE\",\"pin\":2,\"event\":\"HEARTBEAT\",\"state\":\"");
  Serial.print(obstacle ? "DETECTED" : "CLEAR");
  Serial.print("\",\"obstacle\":");
  Serial.print(obstacle ? "true" : "false");
  Serial.print(",\"raw\":");
  Serial.print(rawValue);
  Serial.print(",\"uptime_ms\":");
  Serial.print(millis());
  Serial.println("}");
}
