# CVAE_bc_runner
# franka_rl_bridge — Imitation Learning Extension

ROS 2 package extension for running learned imitation learning policies (BC / ACT+CVAE) on a real Franka robot.

---

## Prerequisites

All prerequisites of the base package must be satisfied first.

---

## Installation

### 1. Install the base package

Clone and set up the base sim2real bridge:
```bash
git clone -b dev https://github.com/janfrischi/ros2-sim2real-bridge.git
```

Follow all installation and build instructions in that repository before continuing.

### 2. Replace the imitation learning module

Replace the existing `imitation_learning` folder inside the base package with the provided one:
```bash
rm -rf ros2-sim2real-bridge/franka_rl_bridge/imitation_learning
cp -r <provided_folder> ros2-sim2real-bridge/franka_rl_bridge/imitation_learning
```

### 3. Rebuild the package
```bash
cd ~/franka_ros2_ws
colcon build --packages-select franka_rl_bridge --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/franka_ros2_ws/install/setup.bash
```

### 4. Install the CVAE Image Publisher

Clone and run the image publisher package:
```bash
git clone https://github.com/mschmotzer/CVAE_Image_publisher.git
```

Follow the run instructions described in that repository. **This node must be running before launching the policy.**

---

## Running

### Step 1 — Start the controller

Launch the Franka controller as described in the [base package](https://github.com/janfrischi/ros2-sim2real-bridge/tree/dev).

### Step 2 — Start the CVAE image publisher

Follow the instructions in the [CVAE Image Publisher repo](https://github.com/mschmotzer/CVAE_Image_publisher).

### Step 3 — Run the policy
```bash
python3 franka_rl_bridge/imitation_learning/bc_policy_runner.py \
  --policy /home/pdz/MasterThesis_MSC/Results_EUler/small_ws_simplified_400/policy_best.ckpt \
  --cvae \
  --context_length 4 \
  --velocity_control \
  --num_cameras 2 \
  --data_norm /home/pdz/MasterThesis_MSC/Results_EUler/small_ws_simplified_400/
```

---

## Adapting to a Different Model

If the model architecture differs (e.g. different embedding length, number of encoder layers, latent dimensions), the following files must be updated manually:

- **`policy_control_cvae.py`** — primary config for inference-time policy parameters
- **`ACTPolicy` class** — architecture definition; update embedding size, encoder depth, or other hyperparameters to match your checkpoint

> ⚠️ Mismatches between the checkpoint architecture and the loaded model definition will cause silent errors or crashes at runtime. Always verify that your config matches the training configuration used to produce the checkpoint.

---

## Arguments Reference

| Argument | Description |
|---|---|
| `--policy` | Path to the trained policy checkpoint (`.ckpt`) |
| `--cvae` | Enable CVAE-based action prediction |
| `--context_length` | Number of past observations used as context |
| `--velocity_control` | Use velocity control mode with position |
| `--num_cameras` | Number of cameras used as input |
| `--data_norm` | Path to directory containing data normalization stats |
