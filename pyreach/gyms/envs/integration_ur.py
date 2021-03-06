# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym environment for the UR Integration Test Workcell."""

import time
from typing import Any, Dict, Tuple, cast

import numpy as np

from pyreach import arm
from pyreach.gyms import arm_element
from pyreach.gyms import core
from pyreach.gyms import reach_env


class IntegrationTestUREnv(reach_env.ReachEnv):
  """Integration Test for UR Environment.

  Intended for connected to the C6A8AD integration testing workcell for
  exercising continuous control on the UR using Reach, PyReach, and Gym.

  """
  SAFE_JOINT_ANGLES: np.ndarray = np.deg2rad([90, -90, 90, -90, 280, 0])

  # pylint: disable=g-bad-todo
  # TODO: Grab MAX and MIN joint angles from real C6A8AD workcell
  MIN_JOINT_ANGLES = SAFE_JOINT_ANGLES - 5.0
  MAX_JOINT_ANGLES = SAFE_JOINT_ANGLES + 5.0

  TIMEOUT_PER_INSTRUCTION_SECONDS = 600.0

  def __init__(self, is_synchronous: bool = True, **kwargs: Any) -> None:
    self.timer_running: bool = False
    self.deadline: float = 0.0
    self.has_fts: bool = True
    response_queue_length: int = 0 if is_synchronous else 2

    pyreach_config: Dict[str, reach_env.ReachElement] = {
        "arm":
            reach_env.ReachArm(
                "",
                tuple(self.MIN_JOINT_ANGLES.tolist()),
                tuple(self.MAX_JOINT_ANGLES.tolist()),
                is_synchronous=is_synchronous,
                response_queue_length=response_queue_length,
                ik_lib=arm.IKLibType.IKFAST,
                e_stop_mode=arm_element.ReachStopMode.STOP_STATUS,
                p_stop_mode=arm_element.ReachStopMode.STOP_STATUS),
        "camera":
            reach_env.ReachColorCamera("realsense", (360, 640)),
        "depth_camera":
            reach_env.ReachDepthCamera("", (720, 1280), color_enabled=True),
        "server":
            reach_env.ReachServer("Server"),
        "vacuum":
            reach_env.ReachVacuum(
                "",
                is_synchronous=is_synchronous,
                state_enable=True,
                vacuum_gauge_enable=True,
                vacuum_detect_enable=True),
        "annotation":
            reach_env.ReachAnnotation(
                "", is_synchronous=False, maximum_size=1024),
    }

    self.has_fts = True
    if "with_fts" in kwargs:
      self.has_fts = cast(bool, kwargs["with_fts"])

    if self.has_fts:
      pyreach_config["fts"] = reach_env.ReachForceTorqueSensor("")

    super().__init__(pyreach_config=pyreach_config, **kwargs)

  def _movej(self,
             joints: np.ndarray) -> Tuple[core.Observation, float, bool, Any]:
    action = {
        "arm": {
            "command": 1,
            "joint_angles": joints,
            "synchronous": 1,
            "velocity": 1.05,
            "acceleration": 1.4
        }
    }
    return super().step(action)

  def _stow_workcell(self) -> core.Observation:
    """Stow the workcell arm clear of the FOV of the camera.

    This will move the workcell arm to a safe joint position.

    Returns:
      Observation after moving arm to safe joint position
    """
    print(f"{time.time():.4f}:ENV: Stowing workcell arm")

    obs, _, _, _ = self._movej(self.SAFE_JOINT_ANGLES)

    return obs

  def reset(self) -> core.Observation:
    """Resets the benchmark.

    Returns:
          Initial observation.
    """
    print(f"{time.time():.4f}:ENV: Resetting the integration test environment")

    # End any current task with reset
    obs = super().reset()
    self.timer_running = False

    return obs

  def close(self) -> None:
    self._stow_workcell()
    print(f"{time.time():.4f}:ENV: Closing gym")
    super().close()

  def step(self,
           action: core.Action) -> Tuple[core.Observation, float, bool, Any]:
    """Perform one step."""
    observation: core.Observation
    reward: float
    done: bool
    info: Any

    observation, reward, done, info = super().step(action)

    if not self.timer_running:
      self.timer_running = True
      self.deadline = time.time() + self.TIMEOUT_PER_INSTRUCTION_SECONDS

    if done:
      self.timer_running = False
      observation = self._stow_workcell()
    elif time.time() >= self.deadline:
      print(f"{time.time():.4f}:ENV: You ran out of time!")
      self.timer_running = False
      reward = -1.0
      done = True
      observation = self._stow_workcell()

    return (observation, reward, done, info)


class IntegrationTestURSyncEnv(IntegrationTestUREnv):
  """Configure a Gym environment with a synchronous arm."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(is_synchronous=True, **kwargs)


class IntegrationTestURAsyncEnv(IntegrationTestUREnv):
  """Configure a Gym environment with an asynchronous arm."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(is_synchronous=False, **kwargs)
