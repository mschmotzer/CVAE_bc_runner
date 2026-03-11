import os
import numpy as np

root_dir = "/home/pdz/MasterThesis_MSC/IsaacLab/datasets/Datasets_benchmarking/1dataset1_no_object_noise.hdf5"

all_positions = []

if os.path.isfile(root_dir) and root_dir.lower().endswith((".h5", ".hdf5")):
    try:
        import h5py
    except ImportError:
        raise ImportError("Please install h5py with: pip install h5py")

    with h5py.File(root_dir, "r") as hf:
        # --- ✅ Handle nested structure under 'data' ---
        if "data" in hf:
            data_group = hf["data"]
        else:
            raise ValueError(f"'data' group not found in {root_dir}")

        demo_names = [k for k in data_group.keys() if k.startswith("demo_")]
        if not demo_names:
            raise ValueError(f"No demo_ groups found inside 'data' in {root_dir}")

        print(f"Found {len(demo_names)} demos under 'data'")

        for demo in sorted(demo_names, key=lambda x: int(x.split("_")[1])):
            joint_path = f"{demo}/initial_state/articulation/robot/joint_position"
            if joint_path not in data_group:
                print(f"⚠️ Warning: joint_position not found for {demo}")
                continue

            arr = np.asarray(data_group[joint_path]).ravel()
            all_positions.append(arr)

else:
    raise FileNotFoundError(f"{root_dir} is not a valid HDF5 file")

# --- Compute average ---
if not all_positions:
    raise ValueError("No joint_position data found!")

lengths = [a.shape[0] for a in all_positions]
if len(set(lengths)) != 1:
    raise ValueError(f"Inconsistent joint vector lengths: {set(lengths)}")

all_positions = np.stack(all_positions, axis=0)
average_joint_position = np.mean(all_positions, axis=0)

print("\n✅ Average Joint Position:")
print(average_joint_position)
