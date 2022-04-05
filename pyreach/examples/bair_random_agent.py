"""
This agent executes a random pick-and-place policy (RAND).
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
from pyreach.tools.analytic_benchmark import random_cloth_pick, random_workspace_point
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent

class BairRandomAgent(BairCollectionAgent):
    MAX_TIMESTEPS = 1000000
    NUM_ROLLOUTS = 100
    MAX_EP_LEN = 100
    LOGFILE = 'logs/{}-true-random.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))
    LOGFREQ = 1 # how often to write logging data to disk
    
    def get_logfile(self):
        return self.LOGFILE

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
        env.set_agent_id("bair-fully-autonomous-agent-v0")
        obs = env.reset()
        assert isinstance(obs, (dict, collections.OrderedDict))

        print(f"{time.time():.4f}:AGENT: Starting data collection.")
        policy_start_time = obs["server"]["latest_ts"]

        logs = []
        rollout = 0
        while rollout < self.NUM_ROLLOUTS:
            print('rollout #{}'.format(rollout))
            self.LOGFILE = self.get_logfile()

            # Get up-to-date observation
            obs, _, done, _ = env.step({})
            assert isinstance(obs, (dict, collections.OrderedDict))

            obs = env.reset()

            env.scramble()
            env.scramble()

            obs = self.stow_arm(env)

            logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
            'act': [], 'full_obs': [obs], 'info': []})

            timestep = 0
            while timestep < self.MAX_EP_LEN:
                img = obs['depth_camera']['color']
                # pick a random pick point on the cloth
                pick_x, pick_y = random_cloth_pick(img)
                # random place point in the workspace
                place_x, place_y = random_workspace_point(pick=(pick_x, pick_y))
                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y)
                except:
                    pick_info = {}
                time.sleep(3)
                obs, _, done, _ = env.step({})

                # log obs and action
                logs[-1]['act'].append([pick_x, pick_y, place_x, place_y])
                logs[-1]['full_obs'].append(obs)
                logs[-1]['cimg'].append(obs['depth_camera']['color'])
                logs[-1]['dimg'].append(obs['depth_camera']['depth'])
                logs[-1]['info'].append(pick_info)

                if timestep % self.LOGFREQ == 0:
                    pickle.dump(logs, open(self.LOGFILE, 'wb'))
                timestep += 1
            rollout += 1
            print(f"{time.time():.4f}:AGENT: An attempt has ended")


def main() -> None:
    agent = BairRandomAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.exec_rollout], [1.0])


if __name__ == "__main__":
    main()
