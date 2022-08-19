import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def train_dqn_model(observation, action_space):
    """
    
    """
    hour = observation[2] # Hour index is 2 for all observations


def dqn_policy(observation, action_space):
    """
    Trained DQN Model.
    """
    hour = observation[2] # Hour index is 2 for all observations
    
    print("--- Building Model ---")
    model = agent.build_model(states, actions)
    print(model.summary())
    print("--- Building Agent ---")
    dqn = agent.build_agent(model, actions)
    print("--- Compiling DQN ---")
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    print("--- Fitting DQN ---")

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    dqn.save_weights('dqn_weights.h5f', overwrite=True)

    
    model = agent.build_model(n_states, n_actions)
    dqn = agent.build_agent(model, n_actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('dqn_weights.h5f')
    _ = dqn.test(env, nb_episodes=5, visualize=False)


class BasicQLearningAgent:
    """
    Basic Q-Learning based agent.
    """
    def __init__(self):
        self.action_space = {}
    
    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        return rbc_policy(observation, action_space)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return rbc_policy(observation, self.action_space[agent_id])


    def build_model(self, states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=states.shape))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(actions), activation='linear'))
        return model



    def build_agent(self, model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                    nb_actions=len(actions), nb_steps_warmup=10, target_model_update=1e-2)
        return dqn

