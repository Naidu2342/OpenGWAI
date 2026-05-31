import h5py
import numpy as np

# ==========================================
# FILE PATH
# ==========================================

file_path = r"C:\Users\Naidu\Downloads\H-H1_LOSC_C00_4_V1-1187006834-4096 (1).hdf5"

# ==========================================
# RECURSIVE EXPLORER
# ==========================================

def explore_item(name, item):

    print("\n" + "=" * 60)

    print(f"PATH: {name}")

    print(f"TYPE: {type(item)}")

    # ======================================
    # DATASET INFO
    # ======================================

    if isinstance(item, h5py.Dataset):

        print(f"SHAPE: {item.shape}")

        print(f"DTYPE: {item.dtype}")

        # Attributes
        if len(item.attrs) > 0:

            print("\nATTRIBUTES:")

            for key, value in item.attrs.items():

                print(f"  {key}: {value}")

        # Sample data
        try:

            data = item[()]

            print("\nSAMPLE VALUES:")

            if np.isscalar(data):

                print(data)

            else:

                flat = np.array(data).flatten()

                print(flat[:20])

        except Exception as e:

            print(f"Could not read dataset values: {e}")

    # ======================================
    # GROUP INFO
    # ======================================

    elif isinstance(item, h5py.Group):

        print("\nGROUP KEYS:")

        for key in item.keys():

            print(f"  - {key}")

        # Attributes
        if len(item.attrs) > 0:

            print("\nATTRIBUTES:")

            for key, value in item.attrs.items():

                print(f"  {key}: {value}")

# ==========================================
# OPEN FILE
# ==========================================

with h5py.File(file_path, "r") as f:

    print("\n" + "#" * 60)

    print("FULL HDF5 FILE EXPLORATION")

    print("#" * 60)

    # File attributes
    if len(f.attrs) > 0:

        print("\nFILE ATTRIBUTES:")

        for key, value in f.attrs.items():

            print(f"{key}: {value}")

    # Recursive traversal
    f.visititems(explore_item)

print("\n" + "#" * 60)

print("EXPLORATION COMPLETE")

print("#" * 60)