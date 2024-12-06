from Reinforcement_Learning.robile_gym_env import RobileEnv

env = RobileEnv()
state = env.reset()
print('state: laser',state[0][0])
next_state = env.step([0.0,0.0,0.0])
print('next_state: laser',next_state[0][0][0])