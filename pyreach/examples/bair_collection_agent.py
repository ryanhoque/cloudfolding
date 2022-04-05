"""
Base agent that the other agents build on which has the implementation of the pick-and-place primitive.
It also has a collect_data function that can used for human data collection, human smoothing rollouts, AEP rollouts, etc
"""
import collections
import time
import cv2
import gym  # type: ignore
import numpy as np  # type: ignore
from pyreach.gyms import core
from pyreach.gyms import experiment_assigner
from pyreach.gyms.envs.benchmark_folding_v2 import BenchmarkFoldingEnv
from pyreach.tools.basic_teleop import get_human_pickplace_action
from pyreach.tools.analytic_benchmark import get_analytic_pick_place_points
from pyreach.tools.bc_smoothing import get_model_prediction
from pyreach.tools.matching_folding import get_folding_actions
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt

class BairCollectionAgent:
    """Agent for collecting BC data or running a trained model."""
    MAX_TIMESTEPS = 1000000
    MAX_EP_LEN = 100
    LOGFILE = 'logs/{}-human-smooth_fold_testing.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))
    LOGFREQ = 1 # how often to write logging data to disk

    def __init__(self, smooth_and_fold=True) -> None:
        # Store anything that needs to be preserved, such as models.
        self._action_id = 0
        self.smoothing = True # Finite State Machine: True if currently smoothing, False if folding.
        self.smooth_and_fold = smooth_and_fold # True if we are doing both smoothing and folding, False for only smoothing


    def pick_and_place(self, env, pick_x, pick_y, place_x, place_y, lay_down=False, only_pick=False):
        """
        Macro action primitive that performs a pick and place by composing smaller actions.
        pick_x, pick_y : source pixel
        place_x, place_y : dest pixel
        lay_down : if True, modify primitive for folding (lower cloth before dropping)
        only_pick : if True, terminate action before translate/release
        Returns: world coordinates x1/y1/z1 and x2/y2 of the pick-place that was performed
        """
        print(f"Executing pick and place action with pick: ({pick_x}, {pick_y}) and place: ({place_x}, {place_y})")
        pick = env.host.depth_camera.image().get_point_normal(pick_x, pick_y)[0]
        place = env.host.depth_camera.image().get_point_normal(place_x, place_y)[0]
        x1, y1, z1 = pick
        x2, y2 = place[:2]

        # Move to safe position
        action = {
            "arm": {
                "joint_angles": env.SAFE_JOINT_ANGLES,
                "command": 1, # joint cmd
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            }
        }
        obs, _, _, _ = env.step(action)
        time.sleep(1)

        # Move to pick point (x1, y1)
        curr_pose = obs['arm']['pose']
        pose = curr_pose.copy()
        pose[0], pose[1] = x1, y1
        pose[3], pose[4], pose[5] = 0.001, -np.pi+0.001, 0.001 #-np.pi/6 + 0.001, -np.pi+0.001, 0.001 #
        action = {
            "arm": {
                "pose": pose,
                "command": 2, # cartesian cmd
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            },
            "vacuum": {
                "state": 0
            }
        }
        obs, _, _, _ = env.step(action)
        time.sleep(1)

        # Lower gripper and grab
        pose[2] = max(z1 - 0.1*(int(lay_down) + 0.2*0.6*int(only_pick)), -0.25) #0.25
        action = {
            "arm": {
                "pose": pose,
                "command": 2,
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            }
        }
        env.step(action)
        time.sleep(1)
        action = {
            "vacuum": {
                "state": 1,
                "synchronous": 1
            }
        }
        env.step(action)
        time.sleep(3)

        pose[2] += 0.005
        action = {
            "arm": {
                "pose": pose,
                "command": 2,
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            }
        }
        env.step(action)
        time.sleep(1)

        # Lift
        pose[2] += 0.1
        action = {
            "arm": {
                "pose": pose,
                "command": 2,
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            }
        }
        env.step(action)
        time.sleep(1)

        if only_pick:
            return {'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2}

        # Translate and release
        pose[0], pose[1] = x2, y2
        action = {
            "arm": {
                "pose": pose,
                "command": 2,
                "synchronous": 1,
                "velocity": 1.04 / (1 + 3*int(lay_down)),
                "acceleration": 1.2
            }
        }
        env.step(action)
        time.sleep(1 + 1*int(lay_down))

        # if lay down, lay down
        if lay_down:
            print("Laying down")
            pose[2] = z1
            action = {
                "arm": {
                    "pose": pose,
                    "command": 2,
                    "synchronous": 1,
                    "velocity": 1.04,
                    "acceleration": 1.2
                }
            }
            env.step(action)
            time.sleep(3)

            # Release vacuum gripper
            action = {
                "vacuum": {
                    "state": 0,
                    "synchronous": 1
                }
            }
            env.step(action)
            time.sleep(2)

            # re-raise gripper
            pose[2] += 0.1
            action = {
                "arm": {
                    "pose": pose,
                    "command": 2,
                    "synchronous": 1,
                    "velocity": 1.04,
                    "acceleration": 1.2
                }
            }
            env.step(action)
            time.sleep(2)
        
        # Release vacuum gripper
        action = {
            "vacuum": {
                "state": 0,
                "synchronous": 1
            }
        }
        env.step(action)
        time.sleep(3)

        # stow arm before taking image of the workspace
        action = {
            "arm": {
                "joint_angles": env.CLEAR_FOV_JOINT_ANGLES,
                "command": 1,
                "synchronous": 1,
                "velocity": 1.04,
                "acceleration": 1.2
            }
        }
        env.step(action)
        return {'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2}

    def collect_data(self, env: BenchmarkFoldingEnv, use_human=True) -> None:
        """Runs the agent loop.

        use_human : if True, solicit human actions through GUI; otherwise compute analytic actions
        """
        env.set_agent_id("bair-collection-agent-v0")

        obs = env.reset()
        assert isinstance(obs, (dict, collections.OrderedDict))

        print(f"{time.time():.4f}:AGENT: Starting data collection.")
        policy_start_time = obs["server"]["latest_ts"]

        timestep = 0
        logs = []
        reset = True

        while timestep < self.MAX_TIMESTEPS:
            # Get up-to-date observation
            obs, _, done, _ = env.step({})
            assert isinstance(obs, (dict, collections.OrderedDict))

            if reset:
                obs = env.reset()
                user_input = input('Scramble, recenter, or neither? s/r/n: ') 
                while user_input.lower() != 'n':
                    if user_input.lower() == 's':
                        env.scramble() 
                    elif user_input.lower() == 'r':
                        action = {
                            "arm": {
                            "joint_angles": env.CLEAR_FOV_JOINT_ANGLES,
                            "command": 1,
                            "synchronous": 1,
                            "velocity": 1.04,
                            "acceleration": 1.2
                            }
                        }
                        env.step(action)
                        time.sleep(2)
                        obs, _, _, _ = env.step({})
                        human_action = get_human_pickplace_action(obs['depth_camera']['color'], analytic_second_point=False)
                        pick_x, pick_y = human_action['pick']
                        place_x, place_y = human_action['place']
                        self.pick_and_place(env, pick_x, pick_y, place_x, place_y)
                        time.sleep(2)
                    user_input = input('Scramble, recenter, or neither? s/r/n: ')
                # stow arm
                action = {
                    "arm": {
                        "joint_angles": env.CLEAR_FOV_JOINT_ANGLES,
                        "command": 1,
                        "synchronous": 1,
                        "velocity": 1.04,
                        "acceleration": 1.2
                    }
                }
                env.step(action)
                time.sleep(2)
                obs, _, _, _ = env.step({})
                logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
                    'act': [], 'full_obs': [obs], 'info': []})
                reset = False

            auto_done_smoothing = False
            auto_done_folding = False
            if use_human:
                if self.smoothing:
                    human_action = get_human_pickplace_action(obs['depth_camera']['color'], analytic_second_point=True)
                else:
                    human_action = get_human_pickplace_action(obs['depth_camera']['color'], analytic_second_point=False)
                pick_x, pick_y = human_action['pick']
                place_x, place_y = human_action['place']
            else: # AEP
                analytic_action = get_analytic_pick_place_points(obs['depth_camera']['color'], env)
                pick_x, pick_y = analytic_action['pick']
                place_x, place_y = analytic_action['place']

            if not auto_done_smoothing:
                pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y)
                pick_info['smoothing'] = int(self.smoothing)
                time.sleep(3)
                obs, _, done, _ = env.step({})

            print("Discard action? y/n") # allow discarding of bad demo actions; after 2 seconds assume we will not discard
            i, o, e = select.select([sys.stdin], [], [], 2)
            if not i or sys.stdin.readline().strip() != 'y':
                # log obs and action 
                logs[-1]['act'].append([pick_x, pick_y, place_x, place_y])
                logs[-1]['full_obs'].append(obs)
                logs[-1]['cimg'].append(obs['depth_camera']['color'])
                logs[-1]['dimg'].append(obs['depth_camera']['depth'])
                logs[-1]['info'].append(pick_info)

            # Logic to check if we are smoothing, folding or done (and should reset)
            if self.smoothing:
                print('Are we done smoothing? y/n')
                i, o, e = select.select([sys.stdin], [], [], 0.1) # timeout after 2 seconds assumes 'no'
                if auto_done_smoothing or (i and sys.stdin.readline().strip() == 'y'):
                    if not self.smooth_and_fold:
                        reset = True # we are not folding, so scramble it
                    else:
                        self.smoothing = False
            else:
                print('Are we done folding? y/n')
                i, o, e = select.select([sys.stdin], [], [], 0,1)
                if auto_done_folding or (i and sys.stdin.readline().strip() == 'y'):
                    reset = True
                    self.smoothing = True

            # Check if environment says we are done. shouldn't happen unless it crosses the massive timeout.
            if done:
                print(f"{time.time():.4f}:AGENT: Step returned done")
                break

            if timestep % self.LOGFREQ == 0:
                # flush logs
                pickle.dump(logs, open(self.LOGFILE, 'wb'))

            timestep += 1

        print(f"{time.time():.4f}:AGENT: An attempt has ended")


def main() -> None:
    agent = BairCollectionAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.collect_data], [1.0])


if __name__ == "__main__":
    main()
