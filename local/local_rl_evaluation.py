import numpy as np
import time
from tensorflow.keras.optimizers import Adam
from agents.dqn_agent import BasicQLearningAgent
import numpy as np
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from citylearn.citylearn import CityLearnEnv
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3.common import env_checker
from pettingzoo.test import api_test

"""
Local Evaluation for Deep RL models. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv

class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation": observations }
    return obs_dict



def train_and_evaluate():
    print("Starting local training and evaluation")
    env = CityLearnEnv(schema=Constants.schema_path)

    # TODO: Add Deep Q Agent here
    #agent = OrderEnforcingAgent()

    # agent = BasicQLearningAgent()


    # actions = ("cooling_storage", "heating_storage", "dhw_storage", "electrical_storage")
    
    model = PPO('MlpPolicy', env, learning_rate=0.001)

    model.learn(total_timesteps=2000000)
    model.save(f'data/models/{model_name}')



    # states = np.array(env.observation_space) # 28 active actions

    # print(states.shape)

    # print("--- Building Model ---")
    # model = agent.build_model(states, actions)
    # print(model.summary())
    # print("--- Building Agent ---")
    # dqn = agent.build_agent(model, actions)
    # print("--- Compiling DQN ---")
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # print("--- Fitting DQN ---")

    # dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # dqn.save_weights('dqn_weights.h5f', overwrite=True)

    
    # model = agent.build_model(n_states, n_actions)
    # dqn = agent.build_agent(model, n_actions)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # dqn.load_weights('dqn_weights.h5f')
    # _ = dqn.test(env, nb_episodes=5, visualize=False)





def evaluate():
    print("Starting local evaluation")
    
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env) # {action_space; observation_space}

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while True:
            observations, _, done, _ = env.step(actions)
            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':
    #evaluate()
    train_and_evaluate()
