import h5py
import numpy as np

# ==========================================
# INPUT / OUTPUT
# ==========================================

INPUT_FILE = r"C:\Users\Naidu\Downloads\H-H1_LOSC_C00_4_V1-1187006834-4096.hdf5"

OUTPUT_FILE = "fake_raw_signal.hdf5"

# ==========================================
# LOAD ORIGINAL STRAIN
# ==========================================

with h5py.File(INPUT_FILE, "r") as f:

    strain = f["strain"]["Strain"][:]

    meta_data = {}

    if "meta" in f:

        for key in f["meta"].keys():

            meta_data[key] = f["meta"][key][()]

# ==========================================
# NORMALIZE
# ==========================================

strain = strain.astype(
    np.float64
)

max_abs = np.max(
    np.abs(strain)
)

strain = strain / max_abs

# ==========================================
# CREATE MASSIVE NOISE
# ==========================================

# ------------------------------------------
# STRONG GAUSSIAN NOISE
# ------------------------------------------

gaussian_noise = np.random.normal(
    0,
    0.5,
    len(strain)
)

# ------------------------------------------
# LOW FREQUENCY DRIFT
# ------------------------------------------

x = np.linspace(
    0,
    100,
    len(strain)
)

drift = (
    0.4
    * np.sin(0.05 * x)
)

# ------------------------------------------
# HIGH FREQUENCY NOISE
# ------------------------------------------

hf_noise = (
    0.2
    * np.sin(40 * x)
)

# ------------------------------------------
# RANDOM GLITCHES
# ------------------------------------------

glitches = np.zeros(
    len(strain)
)

for _ in range(50):

    pos = np.random.randint(
        1000,
        len(strain) - 1000
    )

    width = np.random.randint(
        20,
        200
    )

    amplitude = np.random.uniform(
        0.5,
        2.0
    )

    glitches[
        pos:pos+width
    ] += amplitude

# ==========================================
# COMBINE EVERYTHING
# ==========================================

fake_raw = (
    strain
    + gaussian_noise
    + drift
    + hf_noise
    + glitches
)

# ==========================================
# SAVE NEW HDF5
# ==========================================

with h5py.File(
    OUTPUT_FILE,
    "w"
) as f:

    # --------------------------------------
    # META
    # --------------------------------------

    meta_group = f.create_group(
        "meta"
    )

    for key, value in meta_data.items():

        meta_group.create_dataset(
            key,
            data=value
        )

    # --------------------------------------
    # STRAIN GROUP
    # --------------------------------------

    strain_group = f.create_group(
        "strain"
    )

    ds = strain_group.create_dataset(
        "Strain",
        data=fake_raw
    )

    ds.attrs["Xspacing"] = (
        1.0 / 4096
    )

print("\nFAKE RAW SIGNAL CREATED!")

print(f"\nSaved: {OUTPUT_FILE}")