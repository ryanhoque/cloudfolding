# Cloud Folding

Code for the paper "Learning to Fold Real Garments with One Arm: A Case Study in Cloud-Based Robotics Research" ([website](https://sites.google.com/berkeley.edu/cloudfolding)). This repository is a fork of [Google PyReach](https://github.com/google-research/pyreach). Changes from the parent repository are summarized below.

## Fork Details

Almost all code changes are in the `pyreach/tools` and `pyreach/examples` sub-directories. 

In `pyreach/examples`, we add several agents for the `benchmark_folding_v2` PyReach Gym environment (filenames begin with `bair_`), where each agent executes a different algorithm from the paper (e.g. CRL or KP). These can be run with `python -m pyreach.examples.[filename]` as described in the PyReach README below. 

In `pyreach/tools` we implement code to support the execution of the agent code above. For instance,
- `pyreach/tools/basic_teleop.py` provides a GUI for human pick-and-place teleoperation.
- `corner_pulling.py` implements key parts of the KP algorithm.
- the `pyreach/tools/inv_dynamics/` directory has data processing, model training, and model inference code for the IDYN and CRL algorithms.
- the `pyreach/tools/reach_keypoints/` directory has model training and inference code for training LPAP and KP. This directory is a copy of [hulk-keypoints on Github](https://github.com/jenngrannen/hulk-keypoints), which should be directly cloned from the source to avoid issues if you wish to re-train models for those algorithms.

To run our algorithms on Reach successfully, the following must be true:
- You will need credentials to access Reach, which is currently invite-only.
- You will need to merge upstream changes from [PyReach](https://github.com/google-research/pyreach), since PyReach is constantly updated and this repository is only guaranteed to be compatible with the current version of PyReach. If this leads to nontrivial conflicts, feel free to raise a Github issue and we will try to resolve it.
- You can get our trained models, collected datasets, etc. from the [project website](https://sites.google.com/berkeley.edu/cloudfolding): copy them to the `data/` directory or an otherwise appropriate location for the agents to read them correctly.

The original Google PyReach repository's README follows.

## PyReach README

### Disclaimer: this is not an officially supported Google product.

PyReach is the Python client SDK to remotely control robots.
Operator credentials will be approved by invitation only at this time

## Supported Platform

Ubuntu 18.04, Ubuntu 20.04, and gLinux.

## Build

```shell
git clone https://github.com/google-research/pyreach.git
cd pyreach
./build.sh
```

## Getting Started

If build.sh runs successfully, PyReach is ready to be used. It is supported,
but not required to install PyReach into the system path.

**Step 1.** (Only the first time) Login to the system.

```shell
cd pyreach
./reach login

# Follow the login instructions.
```

**Step 2.** Connect to a robot.

In a new shell session.

```shell
cd pyreach
./reach ls
./reach connect <robot id>
```

**Step 3.** View camera image

In a new shell session.

```shell
cd pyreach
./reach-viewer
```

**Step 4.** Run pendant tool.

In a new shell session.

```shell
cd pyreach
./reach-pendant
```

**Step 5.** Run example agent.

In a new shell session.

```shell
cd pyreach
source setenv.sh
python -m pyreach.examples.pyreach_gym_example
```


**Logs**

By default, all the client logs are saved under:

```shell
$HOME/reach_workspace
```

## Install

Optionally, PyReach can be installed as a pip package:

```shell
cd pyreach
pip install .
```

Once PyReach is installed, the command "reach", "reach-viewer", and "reach-pendant" can be access directly through command line.

## Uninstall

To remove the PyReach pip package:

```shell
pip uninstall pyreach
```
