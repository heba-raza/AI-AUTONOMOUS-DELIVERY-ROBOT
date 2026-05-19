/*
 * ESP32 RC car — WiFi STA + web control
 * Board: ESP32 Dev Module (Arduino-ESP32)
 *
 * RC car joins your router/hotspot WiFi.
 * Open the printed local IP in a browser.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <math.h>

// Pins — match your wiring (L298N-style: INx + ENA/ENB)
const int IN1 = 25;  // Left motor +
const int IN2 = 26;  // Left motor -
const int IN3 = 27;  // Right motor +
const int IN4 = 14;  // Right motor -
const int ENA = 33;  // Left PWM enable
const int ENB = 32;  // Right PWM enable
// Reserved for future I2C (e.g. MPU6050)
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// WiFi STA credentials (set to same network as ESP32-CAM).
// const char *WIFI_SSID = "FAST Faculty";
// const char *WIFI_PASS = "";
const char *WIFI_SSID = "Tenda_D1F218";
const char *WIFI_PASS = "12345678";

WebServer server(80);

// Calibration timings (tune these based on your floor/battery).
unsigned long FORWARD_BURST_MS = 385;
unsigned long TURN_90_MS = 420;

// Motor PWM tuning (0..255).
const int DRIVE_PWM = 220;
const int TURN_ARC_PWM = 170;
const int TURN_PIVOT_PWM = 130;
const int PWM_FREQ = 1000;
const int PWM_RES_BITS = 8;
const float TURN_TARGET_DEG = 85.0f;
const float LEFT_TURN_TRIM_DEG = 6.0f;
const float RIGHT_TURN_TRIM_DEG = 6.0f; // reduce right over-rotation

// MPU6050 (I2C) for gyro-based 90-degree turns.
const uint8_t MPU_ADDR = 0x68;
float gyroZBiasDps = 0.0f;
bool mpuReady = false;
float yawDeg = 0.0f;
float lastGyroZDps = 0.0f;
float lastTurnAngleDeg = 0.0f;
String lastTurnMode = "none";
String motionState = "idle";
String lastCommandResult = "none";
unsigned long imuPrevMicros = 0;

void setMotorPwm(int left, int right) {
  left = constrain(left, 0, 255);
  right = constrain(right, 0, 255);
  ledcWrite(ENA, left);
  ledcWrite(ENB, right);
}

void stopMotors() {
  setMotorPwm(0, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  motionState = "stopped";
}

void driveForward() {
  setMotorPwm(DRIVE_PWM, DRIVE_PWM);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  motionState = "forward";
}

void driveBack() {
  setMotorPwm(DRIVE_PWM, DRIVE_PWM);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  motionState = "back";
}

// Turn left: run right wheel forward, left wheel idle (gentle arc)
void turnLeft() {
  setMotorPwm(0, TURN_ARC_PWM);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  motionState = "left_arc";
}

// Turn right: left forward, right idle
void turnRight() {
  setMotorPwm(TURN_ARC_PWM, 0);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  motionState = "right_arc";
}

// Pivot turns (wheels opposite direction) are more repeatable for angle tests.
void pivotLeft() {
  setMotorPwm(TURN_PIVOT_PWM, TURN_PIVOT_PWM);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  motionState = "left_pivot";
}

void pivotRight() {
  setMotorPwm(TURN_PIVOT_PWM, TURN_PIVOT_PWM);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  motionState = "right_pivot";
}

void runTimed(void (*motion)(), unsigned long ms) {
  motion();
  delay(ms);
  stopMotors();
}

void hardStop(unsigned long brakeMs = 80) {
  // Active brake: short both motor terminals, then release to stop.
  setMotorPwm(255, 255);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, HIGH);
  motionState = "braking";
  delay(brakeMs);
  stopMotors();
}

bool writeMpuReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readMpuBytes(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t got = Wire.requestFrom((int)MPU_ADDR, (int)len, (int)true);
  if (got != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

float readGyroZDps() {
  uint8_t data[2];
  if (!readMpuBytes(0x47, data, 2)) return 0.0f;  // GYRO_ZOUT_H/L
  int16_t raw = (int16_t)((data[0] << 8) | data[1]);
  // +/-250 dps full scale => 131 LSB/(deg/s)
  return ((float)raw / 131.0f) - gyroZBiasDps;
}

bool initMpu6050() {
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(50);
  if (!writeMpuReg(0x6B, 0x00)) return false; // PWR_MGMT_1: wake up
  if (!writeMpuReg(0x1B, 0x00)) return false; // GYRO_CONFIG: +/-250 dps
  if (!writeMpuReg(0x1A, 0x03)) return false; // CONFIG DLPF

  // Estimate Z gyro bias while robot is still.
  const int samples = 300;
  float sum = 0.0f;
  for (int i = 0; i < samples; i++) {
    uint8_t data[2];
    if (!readMpuBytes(0x47, data, 2)) return false;
    int16_t raw = (int16_t)((data[0] << 8) | data[1]);
    sum += (float)raw / 131.0f;
    delay(3);
  }
  gyroZBiasDps = sum / samples;
  return true;
}

void runTurn90WithMpu(bool leftTurn) {
  const float targetDeg = leftTurn
      ? (TURN_TARGET_DEG - LEFT_TURN_TRIM_DEG)
      : (TURN_TARGET_DEG - RIGHT_TURN_TRIM_DEG);
  const unsigned long timeoutMs = 2500;
  const float minUsefulDps = 4.0f;

  float angleDeg = 0.0f;
  unsigned long tPrev = micros();
  unsigned long startMs = millis();

  if (leftTurn) pivotLeft();
  else pivotRight();

  while ((millis() - startMs) < timeoutMs) {
    unsigned long tNow = micros();
    float dt = (float)(tNow - tPrev) / 1000000.0f;
    tPrev = tNow;

    float gz = readGyroZDps();
    if (leftTurn) {
      if (gz < -minUsefulDps) angleDeg += (-gz) * dt;
    } else {
      if (gz > minUsefulDps) angleDeg += gz * dt;
    }
    if (angleDeg >= targetDeg) break;
    delay(5);
  }
  lastTurnAngleDeg = angleDeg;
  lastTurnMode = "gyro";
  stopMotors();
}

void updateImuTelemetry() {
  if (!mpuReady) return;
  unsigned long nowUs = micros();
  if (imuPrevMicros == 0) {
    imuPrevMicros = nowUs;
    return;
  }
  float dt = (float)(nowUs - imuPrevMicros) / 1000000.0f;
  imuPrevMicros = nowUs;
  if (dt <= 0.0f || dt > 0.2f) return;

  float gz = readGyroZDps();
  lastGyroZDps = gz;
  yawDeg += gz * dt;
}

void handleRoot() {
  const char *html = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32 RC</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#1a1a1e;color:#eee;text-align:center;}
  h1{font-size:1.2rem;font-weight:600;}
  .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;max-width:300px;margin:20px auto;}
  button{padding:20px 12px;font-size:1rem;border:none;border-radius:10px;background:#0d9488;color:#fff;cursor:pointer;}
  button:active{opacity:.9;}
  .stop{background:#dc2626;}
  .hint{font-size:.85rem;opacity:.75;margin-top:16px;}
  .tests{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;max-width:300px;margin:8px auto 0;}
  .test{background:#2563eb;}
  .tele{max-width:320px;margin:12px auto 0;padding:10px;border-radius:10px;background:#111827;text-align:left;font-size:.9rem;line-height:1.5;}
</style>
</head>
<body>
<h1>RC Car</h1>
<div class="grid">
  <span></span><button type="button" onclick="go('forward')">Forward</button><span></span>
  <button type="button" onclick="go('left')">Left</button>
  <button type="button" class="stop" onclick="go('stop')">Stop</button>
  <button type="button" onclick="go('right')">Right</button>
  <span></span><button type="button" onclick="go('back')">Back</button><span></span>
</div>
<div class="tests">
  <button class="test" type="button" onclick="testMove('burst')">Forward Burst</button>
  <button class="test" type="button" onclick="testMove('left90')">Left 90</button>
  <button class="test" type="button" onclick="testMove('right90')">Right 90</button>
</div>
<div class="tests">
  <button class="test" type="button" onclick="recalGyro()">Recalibrate Gyro</button>
  <span></span>
  <span></span>
</div>
<div class="tele">
  <div><b>MPU:</b> <span id="mpu">-</span></div>
  <div><b>Yaw (deg):</b> <span id="yaw">-</span></div>
  <div><b>Gyro Z (dps):</b> <span id="gz">-</span></div>
  <div><b>Last turn angle:</b> <span id="turn">-</span></div>
  <div><b>Last turn mode:</b> <span id="mode">-</span></div>
</div>
<p class="hint">If buttons do nothing, check you are on the car WiFi and this page is http://192.168.4.1/</p>
<script>
function go(m){fetch('/cmd?m='+encodeURIComponent(m)).catch(function(){});}
function testMove(name){fetch('/test?name='+encodeURIComponent(name)).catch(function(){});}
function recalGyro(){
  fetch('/gyro/recalibrate')
    .then(function(r){return r.text();})
    .then(function(t){console.log(t);})
    .catch(function(){});
}
function pullTele(){
  fetch('/telemetry')
    .then(r=>r.json())
    .then(t=>{
      document.getElementById('mpu').textContent = t.mpu_ready ? 'READY' : 'NOT FOUND';
      document.getElementById('yaw').textContent = Number(t.yaw_deg).toFixed(2);
      document.getElementById('gz').textContent = Number(t.gyro_z_dps).toFixed(2);
      document.getElementById('turn').textContent = Number(t.last_turn_deg).toFixed(2);
      document.getElementById('mode').textContent = t.last_turn_mode;
    })
    .catch(function(){});
}
setInterval(pullTele, 250);
pullTele();
</script>
</body>
</html>
)HTML";
  server.send(200, "text/html", html);
}

void handleCmd() {
  if (!server.hasArg("m")) {
    server.send(400, "text/plain", "missing m");
    return;
  }
  String m = server.arg("m");
  if (m == "forward") driveForward();
  else if (m == "back") driveBack();
  else if (m == "left") turnLeft();
  else if (m == "right") turnRight();
  else if (m == "stop") stopMotors();
  else if (m == "pause") hardStop(120);
  else {
    server.send(400, "text/plain", "bad cmd");
    return;
  }
  lastCommandResult = m;
  server.send(200, "text/plain", "ok");
}

void handleTest() {
  if (!server.hasArg("name")) {
    server.send(400, "text/plain", "missing name");
    return;
  }
  String name = server.arg("name");
  if (name == "burst") {
    driveForward();
    delay(FORWARD_BURST_MS);
    hardStop(90);
  } else if (name == "backburst") {
    driveBack();
    delay(FORWARD_BURST_MS);
    hardStop(90);
  } else if (name == "left90") {
    // Controls are inverted physically, so map left90 command to right turn.
    if (mpuReady) runTurn90WithMpu(false);
    else {
      runTimed(pivotRight, TURN_90_MS);
      lastTurnAngleDeg = -1.0f;
      lastTurnMode = "timed";
    }
  } else if (name == "right90") {
    // Controls are inverted physically, so map right90 command to left turn.
    if (mpuReady) runTurn90WithMpu(true);
    else {
      runTimed(pivotLeft, TURN_90_MS);
      lastTurnAngleDeg = -1.0f;
      lastTurnMode = "timed";
    }
  } else {
    server.send(400, "text/plain", "bad test");
    return;
  }
  lastCommandResult = String("test:") + name;
  server.send(200, "text/plain", "ok");
}

void handleTelemetry() {
  String s = "{";
  s += "\"mpu_ready\":";
  s += (mpuReady ? "true" : "false");
  s += ",\"motion_state\":\"";
  s += motionState;
  s += "\"";
  s += ",\"yaw_deg\":";
  s += String(yawDeg, 3);
  s += ",\"gyro_z_dps\":";
  s += String(lastGyroZDps, 3);
  s += ",\"last_turn_deg\":";
  s += String(lastTurnAngleDeg, 3);
  s += ",\"last_turn_mode\":\"";
  s += lastTurnMode;
  s += "\"";
  s += ",\"last_command_result\":\"";
  s += lastCommandResult;
  s += "\"}";
  server.send(200, "application/json", s);
}

void handleStatus() {
  handleTelemetry();
}

void handleGyroRecalibrate() {
  stopMotors();
  delay(120);
  bool ok = initMpu6050();
  mpuReady = ok;
  imuPrevMicros = micros();
  lastTurnMode = ok ? "recalibrated" : "recalib_failed";
  lastCommandResult = ok ? "gyro_recalibrated" : "gyro_recalib_failed";
  server.send(ok ? 200 : 500, "text/plain", ok ? "gyro recalibrated" : "gyro recalibration failed");
}

void handlePause() {
  hardStop(120);
  motionState = "paused";
  lastCommandResult = "pause";
  server.send(200, "text/plain", "paused");
}

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  ledcAttach(ENA, PWM_FREQ, PWM_RES_BITS);
  ledcAttach(ENB, PWM_FREQ, PWM_RES_BITS);
  stopMotors();

  mpuReady = initMpu6050();
  Serial.print("MPU6050: ");
  Serial.println(mpuReady ? "READY (gyro turns enabled)" : "NOT FOUND (timed fallback)");
  imuPrevMicros = micros();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi SSID: ");
  Serial.println(WIFI_SSID);
  Serial.print("Open http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");

  server.on("/", HTTP_GET, handleRoot);
  server.on("/cmd", HTTP_GET, handleCmd);
  server.on("/test", HTTP_GET, handleTest);
  server.on("/telemetry", HTTP_GET, handleTelemetry);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/gyro/recalibrate", HTTP_GET, handleGyroRecalibrate);
  server.on("/pause", HTTP_GET, handlePause);
  server.begin();

  Serial.print("Forward burst ms: ");
  Serial.println(FORWARD_BURST_MS);
  Serial.print("Turn 90 ms: ");
  Serial.println(TURN_90_MS);
}

void loop() {
  updateImuTelemetry();
  server.handleClient();
}
