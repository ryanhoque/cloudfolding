"""
KP (keypoints) corner-pulling smoothing agent
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
from pyreach.tools.corner_pulling import get_model_prediction, do_nonmodel_stuff
from pyreach.tools.bc_smoothing import clip
from pyreach.tools.matching_folding import get_folding_actions, get_mask_score
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent

class BairKPAgent(BairCollectionAgent):
    """Agent for collecting BC data or running a trained model."""
    MAX_TIMESTEPS = 1000000
    NUM_ROLLOUTS = 10
    MAX_EP_LEN = 100
    #LOGFILE = 'logs_smooth_fold/{}_smooth-weighted-model-testing.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))
    LOGFREQ = 1 # how often to write logging data to disk
    
    def get_logfile(self):
        return 'logs/{}-keypoints-smooth.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))

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
        env.set_agent_id("bair-kp-agent-v0")
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

            if rollout > 0:
                env.scramble()
                env.scramble()

            self.smoothing = True

            obs = self.stow_arm(env)

            logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
            'act': [], 'full_obs': [obs], 'info': [], 'template_rot':[], 'kps':[]})

            auto_done_smoothing = False
            auto_done_folding = True
            timestep = 0
            max_timestep = 100
            while not (auto_done_smoothing and auto_done_folding) and timestep < max_timestep:
                if not auto_done_smoothing:
                    self.smoothing = True
                    action = get_model_prediction(obs['depth_camera']['color'])
                    force_reset = action.get('force_reset', False)
                    nonmodel_action = do_nonmodel_stuff(obs['depth_camera']['color'])
                    if nonmodel_action is not None:
                        action.update(nonmodel_action)
                    
                    kps = action['kps']
                    print(kps)
                    action_type = action['type']
                    pick_x, pick_y = action['pick']
                    pick_x, pick_y = closest_blue_pixel_to_point(obs['depth_camera']['color'], (pick_y, pick_x), momentum=15)[::-1]
                    place_x, place_y = action['place']
                    pick_y, pick_x = clip((pick_y, pick_x))
                    place_y, place_x = clip((place_y, place_x))
                    match_score = get_mask_score(obs['depth_camera']['color'])
                    auto_done_smoothing = action['done'] or get_num_blue_pixels(obs['depth_camera']['color']) > 44500#and match_score
                    print("coverage thresh:", action['done'], "matching thresh:", match_score,
                        'coverages', action.get('coverage1', None), action.get('coverage2', None))
                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y)
                except:
                    pick_info = {}
                pick_info['smoothing'] = int(self.smoothing)
                time.sleep(3)
                obs, _, done, _ = env.step({})

                # log obs and action
                logs[-1]['act'].append([pick_x, pick_y, place_x, place_y, action_type])
                logs[-1]['full_obs'].append(obs)
                logs[-1]['cimg'].append(obs['depth_camera']['color'])
                logs[-1]['dimg'].append(obs['depth_camera']['depth'])
                logs[-1]['info'].append(pick_info)
                logs[-1]['kps'].append(kps)

                if timestep % self.LOGFREQ == 0:
                    pickle.dump(logs, open(self.LOGFILE, 'wb'))
                timestep += 1

            env.fold_done()
            logs = []
            rollout += 1
            print(f"{time.time():.4f}:AGENT: An attempt has ended")


def main() -> None:
    agent = BairKPAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.exec_rollout], [1.0])


if __name__ == "__main__":
    main()
