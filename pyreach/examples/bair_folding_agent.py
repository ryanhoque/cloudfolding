"""
After human or auto smoothing, this agent executes the folding subtask with either ASM or LPLP.
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
from pyreach.tools.bc_smoothing import get_model_prediction, get_folding_prediction, clip
from pyreach.tools.matching_folding import get_folding_actions, get_mask_score
from pyreach.tools.analytic_benchmark import get_shirt_com, is_too_far, closest_blue_pixel_to_point, get_num_blue_pixels
import pickle
from datetime import datetime
import sys, select
import matplotlib.pyplot as plt
from pyreach.examples.bair_collection_agent import BairCollectionAgent

class BairFoldingAgent(BairCollectionAgent):
    NUM_ROLLOUTS = 5
    MAX_EP_LEN = 100
    LOGFILE = 'logs/{}_fold_AN.pkl'.format(datetime.now().strftime('%m-%d-%H-%M'))
    LOGFREQ = 10 # how often to write logging data to disk
    
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

    def exec_rollout(self, env: BenchmarkFoldingEnv, human_smooth=True, analytic_fold=True) -> None:
        """Runs the agent loop.
        human_smooth: if True, smooth with human teleop; otherwise use LPAP model
        analytic_fold: if True, fold with ASM; otherwise, fold with LPLP model
        """
        # modify agent id to change the logging
        env.set_agent_id("bair-{}-smooth-{}-fold-v1".format('human' if human_smooth else 'auto',
            'ASM' if analytic_fold else 'LPLP'))
        obs = env.reset()
        assert isinstance(obs, (dict, collections.OrderedDict))

        print(f"{time.time():.4f}:AGENT: Starting data collection.")
        policy_start_time = obs["server"]["latest_ts"]

        rollout = 0
        logs = []
        while rollout < self.NUM_ROLLOUTS:
            self.LOGFILE = self.get_logfile()
            print('rollout #{}'.format(rollout))
            folding_actions, fold_count = [], 0

            # Get up-to-date observation
            obs, _, done, _ = env.step({})
            assert isinstance(obs, (dict, collections.OrderedDict))

            obs = env.reset()
            
            user_input = input('Recenter? y/n: ') 
            # allow recentering in case scrambling is missing the folded shirt from the previous rollout.
            if user_input.lower() != 'n':
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
            while not (auto_done_smoothing and auto_done_folding):
                pick_x, pick_y, place_x, place_y = None, None, None, None
                if not auto_done_smoothing:
                    self.smoothing = True
                    if human_smooth:
                        action = get_human_pickplace_action(obs['depth_camera']['color'], analytic_second_point=False)
                    else:
                        action = get_model_prediction(obs['depth_camera']['color'], fully_analytic=False)
                    pick_x, pick_y = action['pick']
                    place_x, place_y = action['place']
                    pick_y, pick_x = clip((pick_y, pick_x))
                    place_y, place_x = clip((place_y, place_x))
                    match_score = get_mask_score(obs['depth_camera']['color'])
                if auto_done_smoothing and not auto_done_folding:
                    self.smoothing = False
                    action_type = 'folding'

                    if fold_count > 3:
                        # Terminate folding rollouts in an open-loop fashion at 4 timesteps
                        auto_done_folding = True
                    else:
                        if analytic_fold:
                            # Execute ASM folding policy
                            folding_actions = get_folding_actions(obs['depth_camera']['color']) if folding_actions == [] else folding_actions
                            pick_x, pick_y = closest_blue_pixel_to_point(obs['depth_camera']['color'], folding_actions[fold_count][0][::-1], momentum=15)[::-1]
                            place_x, place_y = closest_blue_pixel_to_point(obs['depth_camera']['color'], folding_actions[fold_count][1][::-1], momentum=15)[::-1]
                        else:
                            # Execute LPLP folding policy
                            action = get_folding_prediction(obs['depth_camera']['color'])
                            pick_x, pick_y = action['pick']
                            place_x, place_y = action['place']
                            pick_y, pick_x = clip((pick_y, pick_x))
                            place_y, place_x = clip((place_y, place_x))
                        fold_count += 1

                try:
                    pick_info = self.pick_and_place(env, pick_x, pick_y, place_x, place_y, lay_down=auto_done_smoothing)
                except:
                    pick_info = {}
                pick_info['smoothing'] = int(self.smoothing)
                time.sleep(3)
                if human_smooth and not auto_done_smoothing:
                    print('Are we done smoothing? y/n')
                    action_type = 'human_smooth'
                    i, o, e = select.select([sys.stdin], [], [], 2) # timeout after 2 seconds assumes 'no'
                    if (i and sys.stdin.readline().strip() == 'y'):
                        auto_done_smoothing = True
                elif not auto_done_smoothing:
                    auto_done_smoothing = action['done'] or timestep >= 100 #or match_score
                    action_type = action['type']
                    print("coverage thresh:", action['done'], "matching thresh:", match_score,
                        'coverages', action.get('coverage1', None), action.get('coverage2', None))
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
            rollout += 1
            print(f"{time.time():.4f}:AGENT: An attempt has ended")
            pickle.dump(logs, open(self.LOGFILE, 'wb'))

def main() -> None:
    agent = BairFoldingAgent()

    with gym.make("benchmark-folding-v2") as env:
        # To compare multiple agents, pass more than one agents below.
        experiment_assigner.randomized_run(env, [agent.exec_rollout], [1.0])


if __name__ == "__main__":
    main()
