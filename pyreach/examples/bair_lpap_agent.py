"""
This agent executes LPAP, and can also be used for the fully autonomous pipeline A-ASM
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
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent

class BairLPAPAgent(BairCollectionAgent):
    """Agent for collecting BC data or running a trained model."""
    MAX_TIMESTEPS = 1000000
    NUM_ROLLOUTS = 10
    MAX_EP_LEN = 100
    LOGFREQ = 1 # how often to write logging data to disk
    
    def get_logfile(self):
        return 'logs/{}-bc-smoothing-improved-folding-45k-termination.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))

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

    def exec_rollout(self, env: BenchmarkFoldingEnv) -> None:
        """Runs the agent loop.
        """
        # modify agent id to change the logging
        env.set_agent_id("bair-fully-autonomous-agent-v01")
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

            env.scramble()
            env.scramble()
            env.scramble_done()

            self.smoothing = True

            obs = self.stow_arm(env)

            logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
            'act': [], 'full_obs': [obs], 'info': [], 'act_type': []})

            auto_done_smoothing = False
            auto_done_folding = False
            timestep = 0
            while not (auto_done_smoothing and auto_done_folding) and timestep < 100:
                # smooth with LPAP
                pick_x, pick_y, place_x, place_y = None, None, None, None
                if not auto_done_smoothing:
                    self.smoothing = True
                    action = get_model_prediction(obs['depth_camera']['color'], fully_analytic=False)
                    pick_x, pick_y = action['pick']
                    place_x, place_y = action['place']
                    auto_done_smoothing = action['done']
                    action_type = action['type']
                    print("coverage thresh:", action['done'],'coverages', action.get('coverage1', None), action.get('coverage2', None))
                if auto_done_smoothing and not auto_done_folding:
                    # fold with ASM after smoothing
                    self.smoothing = False
                    action_type = 'folding'
                    if fold_count > 3:
                        auto_done_folding = True
                    else:
                        folding_actions = get_folding_actions(obs['depth_camera']['color']) if folding_actions == [] else folding_actions
                        pick_x, pick_y = closest_blue_pixel_to_point(obs['depth_camera']['color'], folding_actions[fold_count][0][::-1], momentum=15)[::-1]
                        place_x, place_y = closest_blue_pixel_to_point(obs['depth_camera']['color'], folding_actions[fold_count][1][::-1], momentum=15)[::-1]
                        fold_count += 1

                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y, lay_down=auto_done_smoothing)
                except:
                    pick_info = {}
                pick_info['smoothing'] = int(self.smoothing)
                time.sleep(3)
                obs, _, done, _ = env.step({})

                # log obs and action
                logs[-1]['act'].append([pick_x, pick_y, place_x, place_y])
                logs[-1]['act_type'].append(action_type)
                logs[-1]['full_obs'].append(obs)
                logs[-1]['cimg'].append(obs['depth_camera']['color'])
                logs[-1]['dimg'].append(obs['depth_camera']['depth'])
                logs[-1]['info'].append(pick_info)

                if timestep % self.LOGFREQ == 0:
                    pickle.dump(logs, open(self.LOGFILE, 'wb'))
                timestep += 1

            env.fold_done()
            logs = []
            rollout += 1
            print(f"{time.time():.4f}:AGENT: An attempt has ended")

def main() -> None:
    agent = BairLPAPAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.exec_rollout], [1.0])


if __name__ == "__main__":
    main()
