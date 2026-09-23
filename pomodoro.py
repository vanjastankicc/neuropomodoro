# -*- coding: utf-8 -*-
"""
pomodoro.py - NeuroPomodoro session

Reads EEG from the Muse 2 over LSL, the model from neuro_train.py says whether
the user is working or resting, and the light changes accordingly
(MQTT -> Home Assistant).

work  - cool white, 70%
nudge - cool white, 100%
break - warm white, 30%
"""

import json
import time

import joblib
import numpy as np
import paho.mqtt.client as mqtt
from pylsl import StreamInlet

import utils as BCIw  # from the NeuroTechX BCI workshop
from muse_stream import start_muse_stream

# ---------------- MQTT ----------------
MQTT_HOST = "192.168.0.31"
MQTT_PORT = 1883
MQTT_USER = "user"
MQTT_PASS = "pswrd"
MQTT_TOPIC = "eeg/light"   # same topic as in the automation

# The bulb is RGB and has no dedicated white LED. In colour temperature mode it
# only reaches 4000 K, which still looks neutral, so white is set as an xy point.
# Roughly 6500 K and 2202 K.
COOL_WHITE = [0.313, 0.329]
WARM_WHITE = [0.507, 0.414]

# (brightness in %, xy)
LIGHT_WORK = (70, COOL_WHITE)
LIGHT_NUDGE = (100, COOL_WHITE)
LIGHT_BREAK = (30, WARM_WHITE)
LIGHT_TRANSITION = 1       # seconds for the fade

# ---------------- Pomodoro ----------------
WORK_TIME = 25 * 60
BREAK_TIME = 5 * 60
REST_TOLERANCE = 15       # seconds of rest before the light brightens
SMOOTH_WINDOW = 5

# ---------------- EEG ----------------
# Must match neuro_train.py, otherwise the model gets different features
BUFFER_LENGTH = 15
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0, 1, 2, 3]   # TP9, AF7, AF8, TP10

LABEL_WORK = 1
LABEL_REST = 0


def create_mqtt_client():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho 2.x
    except AttributeError:
        client = mqtt.Client()                                  # paho 1.x
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    # connect() returns before the broker accepts the login, so the first light
    # command can get lost if we do not wait for it
    start = time.time()
    while not client.is_connected():
        if time.time() - start > 5:
            client.loop_stop()
            raise RuntimeError("Broker refused the connection, check user and password.")
        time.sleep(0.1)
    print("MQTT connected.")
    return client


class LightController:
    def __init__(self, client):
        self.client = client
        self.current = None

    def set(self, setting):
        # no point sending the same state a hundred times a second
        if setting == self.current:
            return
        brightness, xy = setting
        payload = json.dumps({
            "brightness_pct": brightness,
            "xy_color": xy,
            "transition": LIGHT_TRANSITION,
        })
        result = self.client.publish(MQTT_TOPIC, payload)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"Message failed (code {result.rc}).")
            return
        self.current = setting


def connect_to_eeg():
    streams = start_muse_stream()
    inlet = StreamInlet(streams[0], max_chunklen=12)
    fs = int(inlet.info().nominal_srate())
    print(f"Headband connected, {fs} Hz")
    return inlet, fs


def read_chunk(inlet, fs, eeg_buffer, filter_state):
    eeg_data, _ = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
    if not eeg_data:
        return eeg_buffer, filter_state, False
    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
    # the notch filter removes mains interference
    eeg_buffer, filter_state = BCIw.update_buffer(
        eeg_buffer, ch_data, notch=True, filter_state=filter_state
    )
    return eeg_buffer, filter_state, True


def dominant_state(buffer):
    if not buffer:
        return None
    # on a tie it stays on work, so the light does not brighten without reason
    return LABEL_WORK if buffer.count(LABEL_WORK) >= buffer.count(LABEL_REST) else LABEL_REST


# ---------------- setup ----------------
mqtt_client = create_mqtt_client()
light = LightController(mqtt_client)

inlet, fs = connect_to_eeg()
clf = joblib.load("mental_model.pkl")
print("Model loaded.")

eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNEL)))
filter_state = None
smooth_max_len = int(SMOOTH_WINDOW / SHIFT_LENGTH)

# ---------------- main loop ----------------
try:
    while True:

        # --- work block ---
        print("\nWORK BLOCK")
        light.set(LIGHT_WORK)

        smooth_buffer = []
        rest_start = None
        work_start = time.time()

        while time.time() - work_start < WORK_TIME:
            eeg_buffer, filter_state, ok = read_chunk(inlet, fs, eeg_buffer, filter_state)
            if not ok:
                continue

            epoch = BCIw.get_last_data(eeg_buffer, int(EPOCH_LENGTH * fs))
            feat_vector = BCIw.compute_feature_vector(epoch, fs)
            pred = clf.predict([feat_vector])[0]

            # a single prediction is unreliable, so take whatever dominates
            # the last 5 seconds
            smooth_buffer.append(pred)
            if len(smooth_buffer) > smooth_max_len:
                smooth_buffer.pop(0)
            state = dominant_state(smooth_buffer)

            if state == LABEL_WORK:
                rest_start = None
                light.set(LIGHT_WORK)
                print("WORK")
            else:
                if rest_start is None:
                    rest_start = time.time()
                resting_for = time.time() - rest_start
                print(f"REST {resting_for:.1f} s")

                if resting_for >= REST_TOLERANCE:
                    if light.current != LIGHT_NUDGE:
                        print("Resting too long, brightening the light.")
                    light.set(LIGHT_NUDGE)

        # --- break ---
        print("\nBREAK")
        light.set(LIGHT_BREAK)

        # keep reading the stream so the connection stays alive,
        # but classify nothing
        break_start = time.time()
        while time.time() - break_start < BREAK_TIME:
            eeg_buffer, filter_state, _ = read_chunk(inlet, fs, eeg_buffer, filter_state)

        print("Break over.")

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()