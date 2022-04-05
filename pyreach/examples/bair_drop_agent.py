"""
This agent demonstrates a COM drop action as used in the DROP algorithm
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
from pyreach.tools.matching_folding import get_folding_actions, get_mask_score
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels, get_boundary_pick_point
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent

WORKSPACE_X_BOUNDS = (130, 540)
WORKSPACE_Y_BOUNDS = (40, 270)

class BairDropAgent(BairCollectionAgent):
    """Agent for collecting BC data or running a trained model."""
    MAX_TIMESTEPS = 1000000
    NUM_ROLLOUTS = 10
    MAX_EP_LEN = 100
    LOGFREQ = 1 # how often to write logging data to disk
    
    def get_logfile(self):
        return 'logs/{}-com-pick-dynamic-layout_v2.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))

    def stow_arm(self, env):
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
        return obs

    def collect_data(self, env: BenchmarkFoldingEnv) -> None:
        """Runs the agent loop.
        """
        # modify agent id to change the logging
        env.set_agent_id("bair-fully-autonomous-agent-v0")
        obs = env.reset()
        assert isinstance(obs, (dict, collections.OrderedDict))

        print(f"{time.time():.4f}:AGENT: Starting data collection.")
        policy_start_time = obs["server"]["latest_ts"]

        rollout = 0
        while rollout < self.NUM_ROLLOUTS:
            self.LOGFILE = self.get_logfile()
            folding_actions, fold_count = [], 0
            logs = []

            # Get up-to-date observation
            obs, _, done, _ = env.step({})
            assert isinstance(obs, (dict, collections.OrderedDict))

            obs = env.reset()
            # env.scramble()
            self.smoothing = True
            obs = self.stow_arm(env)

            logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
            'act': [], 'full_obs': [obs], 'info': []})

            auto_done_smoothing = False
            auto_done_folding = False
            timestep = 0
            while timestep < 200:
                env.scramble(1)
                obs = self.stow_arm(env)

                pick_info = {}
                # execute a COM drop action...
                pick_y, pick_x = get_shirt_com(obs['depth_camera']['color']) 
                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, pick_x, pick_y, only_pick=True)
                    time.sleep(2)
                except:
                    pick_info = {}

                # lift to drop height
                drop_height = {
                    "arm": {
                        "command": 2,
                        "pose": env.CENTER_LIFT_POSE,
                        "synchronous": 1,
                        "velocity": 1.04,
                        "acceleration": 1.2
                    }
                }
                env.step(drop_height)
                time.sleep(3)
                
                # open gripper
                action = {"vacuum": {"state": 0}}
                env.step(action)
                time.sleep(0.25)
                
                pick_info['smoothing'] = int(self.smoothing)

                obs = self.stow_arm(env)
                obs, _, done, _ = env.step({})

                # log obs and action
                logs[-1]['act'].append([pick_x, pick_y])
                logs[-1]['full_obs'].append(obs)
                logs[-1]['cimg'].append(obs['depth_camera']['color'])
                logs[-1]['dimg'].append(obs['depth_camera']['depth'])
                logs[-1]['info'].append(pick_info)

                if timestep % self.LOGFREQ == 0:
                    pickle.dump(logs, open(self.LOGFILE, 'wb'))
                timestep += 1

            print(f"{time.time():.4f}:AGENT: An attempt has ended")

def main() -> None:
    agent = BairDropAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.collect_data], [1.0])


if __name__ == "__main__":
    main()
