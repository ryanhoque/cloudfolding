"""
This agent executes either inverse dynamics IDYN or CRL, which can be interpreted as forward dynamics.
"""
import collections
import time
import cv2
import gym  # type: ignore
import numpy as np  # type: ignore
from pyreach.gyms import core
from pyreach.gyms import experiment_assigner
from pyreach.tools.bc_smoothing import clip, do_nonmodel_stuff
from pyreach.gyms.envs.benchmark_folding_v2 import BenchmarkFoldingEnv
from pyreach.tools.matching_folding import get_folding_actions, get_mask_score
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent
from pyreach.tools.inv_dynamics.test import InvInference, ForwardInference

class BairDynAgent(BairCollectionAgent):
    MAX_TIMESTEPS = 1000000
    NUM_ROLLOUTS = 5
    MAX_EP_LEN = 100
    LOGFREQ = 20 # how often to write logging data to disk
    DATADIR = 'data/'
    
    def __init__(self, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.LOGFILE = 'logs/{}-{}-dynamics.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'), 
                'inverse' if inverse else 'forward')

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
        if self.inverse:
            env.set_agent_id("bair-invdyn-agent-v0")
        else:
            env.set_agent_id("bair-forward-dyn-agent-v0")
        obs = env.reset()
        assert isinstance(obs, (dict, collections.OrderedDict))

        print(f"{time.time():.4f}:AGENT: Starting data collection.")
        policy_start_time = obs["server"]["latest_ts"]

        rollout = 0
        logs = []
        if self.inverse:
            inf = InvInference(DATADIR+'IDYN.pth', device_idx=0)
        else:
            inf = ForwardInference(DATADIR+'CRL.pth', device_idx=0)
        smooth_goal = np.load(DATADIR+'smooth.npy')

        while rollout < self.NUM_ROLLOUTS:

            # Get up-to-date observation
            print("rollout #{}".format(rollout))
            obs, _, done, _ = env.step({})
            assert isinstance(obs, (dict, collections.OrderedDict))

            obs = env.reset()

            env.scramble()
            env.scramble()

            self.smoothing = True

            obs = self.stow_arm(env)

            logs.append({'cimg': [obs['depth_camera']['color']], 'dimg': [obs['depth_camera']['depth']], 
            'act': [], 'full_obs': [obs], 'info': []})

            auto_done_smoothing = False
            auto_done_folding = False
            timestep = 0
            while not auto_done_smoothing and timestep < self.MAX_EP_LEN:
                if not auto_done_smoothing:
                    obs = obs['depth_camera']['color']
                    action = do_nonmodel_stuff(obs)
                    act_uses_model = False
                    if action is None:
                        act_uses_model = True
                        print('running inference...')
                        if self.inverse:
                            action = inf.predict(obs, smooth_goal, closest=False)
                        else:
                            action = inf.random_sampling(obs)
                    pick_x, pick_y = action['pick']
                    pick_yc, pick_xc = clip(closest_blue_pixel_to_point(obs, (pick_y, pick_x), momentum=15))
                    place_y, place_x = clip(action['place'][::-1])
                    auto_done_smoothing = get_num_blue_pixels(obs) > 45000
                    pick_x, pick_y = pick_xc, pick_yc
                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y)
                except:
                    pick_info = {}
                pick_info['smoothing'] = int(self.smoothing)
                pick_info['act_uses_model'] = act_uses_model # is this from model output or recenter/random?
                pick_info['actual_pick'] = action.copy() # w/o jump to nearest blue pixel
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

            #env.fold_done()
            rollout += 1
            print(f"{time.time():.4f}:AGENT: An attempt has ended")
        pickle.dump(logs, open(self.LOGFILE, 'wb'))
        sys.exit(0)


def main() -> None:
    agent = BairDynAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        # Ryan: this function can support multithreading, but we don't need that right now.
        experiment_assigner.randomized_run(env, [agent.exec_rollout], [1.0])


if __name__ == "__main__":
    main()
