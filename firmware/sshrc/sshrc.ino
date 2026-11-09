#include <Arduino.h>
#include "src/config.h"
#include "src/classifier.h"
#include "src/features_dev.h"
#include "src/decision.h"

// Number of values per second of signal, per channel (ACC has 3 axes).
#define PER_S_BVP FS_BVP
#define PER_S_EDA FS_EDA
#define PER_S_TEMP FS_TEMP
#define PER_S_ACC (FS_ACC * 3)
#define N_PER_S (PER_S_BVP + PER_S_EDA + PER_S_TEMP + PER_S_ACC)

// Incoming text line from the host (one command per line).
static char line[4096];
static size_t len = 0;

// 30 s ring buffers, one slot per second
static float rb_bvp[WINDOW_S * PER_S_BVP], rb_eda[WINDOW_S * PER_S_EDA], rb_temp[WINDOW_S * PER_S_TEMP];
static int8_t rb_acc[WINDOW_S * PER_S_ACC];
static float lin_bvp[WIN_N_BVP], lin_eda[WIN_N_EDA], lin_temp[WIN_N_TEMP];
static int8_t lin_acc[WIN_N_ACC * 3];
static int slot = 0;          // next slot to write = oldest slot once full
static long n_sec = 0;
static bool operational = false;   // true: only "A" / "D,<frame>" replies to "S"

// Boot banner: firmware version, feature counts, which subjects the model excluded, free memory.
static void ready_line() {
  Serial.printf("READY sshrc v4 n_feat=%d n_dev_feat=%d model_excluded=%s free_heap=%u\n", N_FEAT, N_DEV_FEAT,
                MODEL_EXCLUDED, (unsigned)ESP.getFreeHeap());
}

// 'G' command: classify one golden feature vector and report the int32 result and the time taken.
static void golden(char *p) {
  long idx = strtol(p, &p, 10);
  float x[N_FEAT];
  for (int i = 0; i < N_FEAT; i++) {
    if (*p != ',') { Serial.printf("E,%ld,missing feature %d\n", idx, i); return; }
    p++;
    x[i] = strtof(p, &p);
  }
  float logit;
  int64_t t0 = esp_timer_get_time();
  int32_t acc = sshrc_classify(x, &logit);
  int64_t t1 = esp_timer_get_time();
  Serial.printf("R,%ld,%ld,%d,%.6f,%lld\n", idx, (long)acc, acc > 0 ? 1 : 0, logit, (long long)(t1 - t0));
}

// 'S' command: store one second of raw signal in the ring buffer. Every HOP_S seconds (once
// WINDOW_S seconds are buffered) run features -> decision -> frame for the last 30 s.
static void second(char *p) {
  float *b = rb_bvp + slot * PER_S_BVP, *e = rb_eda + slot * PER_S_EDA, *t = rb_temp + slot * PER_S_TEMP;
  int8_t *a = rb_acc + slot * PER_S_ACC;
  for (int i = 0; i < N_PER_S; i++) {
    if (*p != ',') { Serial.printf("E,second %ld: %d of %d values\n", n_sec, i, N_PER_S); return; }
    p++;
    float v = strtof(p, &p);
    if (i < PER_S_BVP) b[i] = v;
    else if (i < PER_S_BVP + PER_S_EDA) e[i - PER_S_BVP] = v;
    else if (i < PER_S_BVP + PER_S_EDA + PER_S_TEMP) t[i - PER_S_BVP - PER_S_EDA] = v;
    else a[i - PER_S_BVP - PER_S_EDA - PER_S_TEMP] = (int8_t)lrintf(v);
  }
  // Advance the ring buffer; only every HOP_S-th second produces a window.
  slot = (slot + 1) % WINDOW_S;
  n_sec++;
  if (n_sec < WINDOW_S || (n_sec - WINDOW_S) % HOP_S != 0) {
    if (operational) Serial.println("A"); else Serial.printf("A,%ld\n", n_sec);
    return;
  }

  int64_t w0 = esp_timer_get_time();
  for (int k = 0; k < WINDOW_S; k++) {          // unroll the ring, oldest second first
    int s = (slot + k) % WINDOW_S;
    memcpy(lin_bvp + k * PER_S_BVP, rb_bvp + s * PER_S_BVP, PER_S_BVP * sizeof(float));
    memcpy(lin_eda + k * PER_S_EDA, rb_eda + s * PER_S_EDA, PER_S_EDA * sizeof(float));
    memcpy(lin_temp + k * PER_S_TEMP, rb_temp + s * PER_S_TEMP, PER_S_TEMP * sizeof(float));
    memcpy(lin_acc + k * PER_S_ACC, rb_acc + s * PER_S_ACC, PER_S_ACC);
  }
  float f[N_DEV_FEAT];
  StageTimes st;
  // 25 features and quality metrics of the window (features_dev.cpp).
  dev_features(lin_bvp, lin_eda, lin_temp, lin_acc, f, &st);
  int64_t w1 = esp_timer_get_time();
  DecOut d;
  // Quality flag, calibration, classifier, effort gate, state machine, 8-byte frame (decision.cpp).
  dec_step(f, &d);
  int64_t w2 = esp_timer_get_time();
  digitalWrite(LED_PIN, (d.state == 2 || d.fallback) ? HIGH : LOW);
  if (operational) {                              // privacy boundary: the frame and nothing else
    Serial.print("D,");
    for (int i = 0; i < 8; i++) Serial.printf("%02x", d.frame[i]);
    Serial.println();
    return;
  }
  // Debug/replay mode: also report all features, timings and the decision (used by the test scripts).
  Serial.printf("F,%ld", n_sec);
  for (int i = 0; i < N_DEV_FEAT; i++) Serial.printf(",%.9g", f[i]);
  Serial.printf(",%ld,%ld,%ld,%ld,%ld,%ld,%lld,", (long)st.bvp_filter_us, (long)st.bvp_beats_us, (long)st.eda_us,
                (long)st.temp_us, (long)st.acc_us, (long)st.total_us, (long long)(w1 - w0));
  for (int i = 0; i < 8; i++) Serial.printf("%02x", d.frame[i]);
  Serial.printf(",%ld,%.9g,%d,%.9g,%d,%d,%d,%d,%lld\n", (long)d.acc, d.p, d.state, d.conf, d.qflag, d.effort,
                d.fallback, d.reason, (long long)(w2 - w1));
}

// Dispatches one command line (see the protocol at the top of this file).
static void handle(char *s) {
  if (s[0] == 'P' && s[1] == 0) { ready_line(); return; }
  if (s[0] == 'N' && s[1] == 0) {
    Serial.print("NAMES");
    for (int i = 0; i < N_DEV_FEAT; i++) { Serial.print(','); Serial.print(DEV_FEAT_NAMES[i]); }
    Serial.println();
    return;
  }
  if (s[0] == 'R' && s[1] == 0) { slot = 0; n_sec = 0; Serial.println("OK,reset"); return; }
  if (s[0] == 'C' && s[1] == 0) { slot = 0; n_sec = 0; dec_new_session(); Serial.println("OK,session"); return; }
  if (s[0] == 'O' && s[1] == 0) { operational = true; Serial.println("OK,operational"); return; }
  if (s[0] == 'M' && s[1] == 0) {
    Serial.printf("M,free_heap=%u,min_free_heap=%u,max_alloc=%u,stack_hwm=%u\n", (unsigned)ESP.getFreeHeap(),
                  (unsigned)ESP.getMinFreeHeap(), (unsigned)ESP.getMaxAllocHeap(),
                  (unsigned)uxTaskGetStackHighWaterMark(NULL));
    return;
  }
  if (s[0] == 'G' && s[1] == ',') { golden(s + 2); return; }
  if (s[0] == 'S' && s[1] == ',') { second(s + 1); return; }
  Serial.println("E,unknown command");
}

// Runs once at power-on: serial link, LED, empty session, READY line.
void setup() {
  Serial.setRxBufferSize(8192);
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  delay(300);
  dec_new_session();
  ready_line();
}

// Reads characters from the serial port and handles each complete line.
void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') { line[len] = 0; if (len) handle(line); len = 0; }
    else if (len < sizeof(line) - 1) line[len++] = c;
    else { len = 0; Serial.println("E,line too long"); }
  }
}
