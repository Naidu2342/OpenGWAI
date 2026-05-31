from pathlib import Path

import h5py
import numpy as np

# ==========================================
# SETTINGS
# ==========================================

DATA_FOLDER = r"C:\Users\Naidu\Downloads\datasets"

# Smaller windows = more training samples
WINDOW_SIZE = 1024

# Overlapping windows
STEP_SIZE = 256

OUTPUT_X = "X.npy"
OUTPUT_Y = "y.npy"

SAMPLE_RATE = 4096

# ==========================================
# EVENT WINDOWS
# ==========================================

EVENT_WINDOWS = {

    # GW150914
    "GW150914": {
        "positive_start": 14,
        "positive_end": 18
    },

    # GW170817
    "GW170817": {
        "positive_start": 1000,
        "positive_end": 1010
    }
}

# ==========================================
# HELPERS
# ==========================================

def normalize_signal(signal):

    signal = signal.astype(np.float64)

    max_abs = np.max(np.abs(signal))

    if max_abs == 0:
        return signal

    return signal / max_abs


def extract_windows(
    signal,
    start_index,
    end_index,
    label
):

    global X
    global y

    for start in range(
        start_index,
        end_index - WINDOW_SIZE,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE

        chunk = signal[start:end]

        if len(chunk) != WINDOW_SIZE:
            continue

        # Skip invalid chunks
        if np.any(np.isnan(chunk)):
            continue

        if np.any(np.isinf(chunk)):
            continue

        X.append(chunk)

        y.append(label)

# ==========================================
# BUILD DATASET
# ==========================================

X = []
y = []

files = list(
    Path(DATA_FOLDER).glob("*.hdf5")
)

print("\nFound files:")

for file in files:
    print(file.name)

# ==========================================
# PROCESS FILES
# ==========================================

for file in files:

    print("\n" + "=" * 60)

    print(f"Processing: {file.name}")

    # ======================================
    # DETECT EVENT TYPE
    # ======================================

    if "170817" in file.name:

        event_key = "GW170817"

    else:

        event_key = "GW150914"

    print(f"Detected Event: {event_key}")

    # ======================================
    # LOAD SIGNAL
    # ======================================

    with h5py.File(file, "r") as f:

        strain = f["strain"]["Strain"][:]

    strain = np.asarray(
        strain,
        dtype=np.float64
    )

    print(f"Signal Shape: {strain.shape}")

    # ======================================
    # NORMALIZE
    # ======================================

    strain = normalize_signal(strain)

    # ======================================
    # EVENT REGIONS
    # ======================================

    pos_start = int(
        EVENT_WINDOWS[event_key]["positive_start"]
        * SAMPLE_RATE
    )

    pos_end = int(
        EVENT_WINDOWS[event_key]["positive_end"]
        * SAMPLE_RATE
    )

    print(f"Positive Region: {pos_start} → {pos_end}")

    # ======================================
    # POSITIVE SAMPLES
    # ======================================

    print("\nExtracting POSITIVE windows...")

    before_positive = len(X)

    extract_windows(
        signal=strain,
        start_index=pos_start,
        end_index=pos_end,
        label=1
    )

    positive_added = len(X) - before_positive

    print(f"Positive Windows Added: {positive_added}")

    # ======================================
    # NEGATIVE REGION 1
    # ======================================

    print("\nExtracting NEGATIVE windows (before event)...")

    before_negative = len(X)

    negative_region_1_end = max(
        pos_start - (10 * SAMPLE_RATE),
        WINDOW_SIZE
    )

    extract_windows(
        signal=strain,
        start_index=0,
        end_index=negative_region_1_end,
        label=0
    )

    negative_1_added = len(X) - before_negative

    print(f"Negative Windows Added: {negative_1_added}")

    # ======================================
    # NEGATIVE REGION 2
    # ======================================

    print("\nExtracting NEGATIVE windows (after event)...")

    before_negative_2 = len(X)

    negative_region_2_start = min(
        pos_end + (10 * SAMPLE_RATE),
        len(strain) - WINDOW_SIZE
    )

    extract_windows(
        signal=strain,
        start_index=negative_region_2_start,
        end_index=len(strain),
        label=0
    )

    negative_2_added = len(X) - before_negative_2

    print(f"Negative Windows Added: {negative_2_added}")

# ==========================================
# CONVERT TO ARRAYS
# ==========================================

print("\n" + "=" * 60)

print("Converting to NumPy arrays...")

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)

# ==========================================
# SHUFFLE DATASET
# ==========================================

print("\nShuffling dataset...")

indices = np.arange(len(X))

np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# ==========================================
# DATASET INFO
# ==========================================

print("\n" + "=" * 60)

print("FINAL DATASET")

print("=" * 60)

print(f"\nX shape: {X.shape}")

print(f"y shape: {y.shape}")

noise_count = np.sum(y == 0)

wave_count = np.sum(y == 1)

print("\nClass Distribution:")

print(f"Noise Samples: {noise_count}")

print(f"Wave Samples: {wave_count}")

balance_ratio = (
    noise_count / wave_count
    if wave_count > 0
    else 0
)

print(f"\nImbalance Ratio: {balance_ratio:.2f}:1")

# ==========================================
# SAVE DATASET
# ==========================================

print("\nSaving dataset...")

np.save(OUTPUT_X, X)

np.save(OUTPUT_Y, y)

print("\nDataset Saved Successfully!")

print(f"\nSaved:")
print(f"- {OUTPUT_X}")
print(f"- {OUTPUT_Y}")

print("\nDONE")