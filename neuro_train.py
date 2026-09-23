# -*- coding: utf-8 -*-
"""
neuro_train.py - calibration for the NeuroPomodoro system

Records EEG from the Muse 2 over LSL while the user alternates between doing
their own task and sitting still, computes band-power features and trains an
SVM for that user (1 = work, 0 = rest).

Windows overlap by 80%, so a random train/test split would scatter nearly
identical windows into both sets and the accuracy would come out better than
it really is. Testing is therefore done by cycle: one cycle is left out, the
model is trained on the rest and tested on the held-out one. The final model
is trained on everything.

Output:
    mental_model.pkl   - trained model (StandardScaler + SVM)
    work.npy, rest.npy - raw feature vectors
    cycles_work.npy, cycles_rest.npy - which cycle each vector came from
"""

import time
import winsound

import joblib
import numpy as np
from pylsl import StreamInlet
from sklearn import svm
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils as BCIw  # from the NeuroTechX BCI workshop
from muse_stream import start_muse_stream

# ---------------- settings ----------------
REST_DURATION = 30       # seconds per rest block
WORK_DURATION = 30       # seconds per work block
CYCLES = 6               # about six minutes in total
WARMUP_DURATION = 3      # this much is thrown away at the start

BUFFER_LENGTH = 15       # seconds of EEG held in the buffer
EPOCH_LENGTH = 1         # window length (s)
OVERLAP_LENGTH = 0.8     # overlap between consecutive windows (s)
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0, 1, 2, 3]   # TP9, AF7, AF8, TP10

LABEL_WORK = 1
LABEL_REST = 0


def beep():
    # tells the user it is time to switch state
    winsound.Beep(1000, 200)


def connect_to_eeg():
    streams = start_muse_stream()
    inlet = StreamInlet(streams[0])
    fs = int(inlet.info().nominal_srate())
    print(f"Headband connected, {fs} Hz, "
          f"channels: {inlet.info().channel_count()}")
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


def record_block(inlet, fs, duration, eeg_buffer, filter_state):
    # records one block and returns every feature vector from it
    features = []
    start_time = time.time()
    while time.time() - start_time < duration:
        eeg_buffer, filter_state, ok = read_chunk(inlet, fs, eeg_buffer, filter_state)
        if not ok:
            continue
        epoch = BCIw.get_last_data(eeg_buffer, int(EPOCH_LENGTH * fs))
        features.append(BCIw.compute_feature_vector(epoch, fs))
    return features, eeg_buffer, filter_state


# ---------------- recording ----------------
inlet, fs = connect_to_eeg()
eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), len(INDEX_CHANNEL)))
filter_state = None

# Fill the buffer with real data first, so the first windows are not zeros
# or the transient response of the filter.
print(f"Warming up for {WARMUP_DURATION} s...")
warmup_start = time.time()
while time.time() - warmup_start < WARMUP_DURATION:
    eeg_buffer, filter_state, _ = read_chunk(inlet, fs, eeg_buffer, filter_state)

work_data, rest_data = [], []
work_cycles, rest_cycles = [], []

print("\nCALIBRATION")
print(f"{CYCLES} cycles, rest then work "
      f"({REST_DURATION} s / {WORK_DURATION} s)")

try:
    for cycle in range(CYCLES):
        # Eyes alternate between cycles, so the model does not learn the
        # difference between open and closed eyes instead of the states.
        eyes = "eyes closed" if cycle % 2 == 0 else "eyes open"
        print(f"\nCycle {cycle + 1}/{CYCLES}: REST, {eyes}")
        beep()
        feats, eeg_buffer, filter_state = record_block(
            inlet, fs, REST_DURATION, eeg_buffer, filter_state
        )
        rest_data.extend(feats)
        # remember which cycle each window came from, needed for validation
        rest_cycles.extend([cycle] * len(feats))

        print(f"Cycle {cycle + 1}/{CYCLES}: WORK")
        beep()
        feats, eeg_buffer, filter_state = record_block(
            inlet, fs, WORK_DURATION, eeg_buffer, filter_state
        )
        work_data.extend(feats)
        work_cycles.extend([cycle] * len(feats))

except KeyboardInterrupt:
    print("\nRecording stopped.")

# ---------------- save raw features ----------------
X_work = np.array(work_data)
X_rest = np.array(rest_data)

np.save("work.npy", X_work)
np.save("rest.npy", X_rest)
np.save("cycles_work.npy", np.array(work_cycles))
np.save("cycles_rest.npy", np.array(rest_cycles))

X = np.vstack((X_work, X_rest))
y = np.hstack((np.full(len(X_work), LABEL_WORK),
               np.full(len(X_rest), LABEL_REST)))
groups = np.hstack((work_cycles, rest_cycles))

print(f"\nCollected {len(X_work)} work and {len(X_rest)} rest windows.")


def build_model():
    # standardisation, then an SVM with an RBF kernel
    return make_pipeline(StandardScaler(), svm.SVC(kernel="rbf"))


# ---------------- accuracy check ----------------
# The Pipeline goes into cross_val_score, so the scaler is fitted only on the
# training cycles and never on the one being tested.
n_complete_cycles = len(np.unique(groups))
if n_complete_cycles >= 2:
    scores = cross_val_score(build_model(), X, y,
                             groups=groups, cv=LeaveOneGroupOut())
    for i, score in enumerate(scores):
        print(f"Cycle {i + 1} held out: accuracy {score:.2f}")
    print(f"Mean accuracy: {scores.mean():.2f} (SD {scores.std():.2f})")
else:
    print("At least two cycles are needed for the check.")

# ---------------- final model ----------------
clf = build_model()
clf.fit(X, y)
joblib.dump(clf, "mental_model.pkl")
print("\nModel saved as 'mental_model.pkl'")