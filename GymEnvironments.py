from gym import envs
import gym_pcgrl

[print(env_id) for env_id, env in envs.registry.items() if "gym_pcgrl" in env.entry_point]
# for env_id, env in envs.registry.items():
#     print(env)
#     print(env.entry_point)
#     print("\n")


# Sokoban environment with Narrow representation:
#
# import gym
# import gym_pcgrl
#
# env = gym.make('sokoban-narrow-v0')
# obs = env.reset()
# for t in range(1000):
#   action = env.action_space.sample()
#   obs, reward, done, info = env.step(action)
#   env.render('human')
#   if done:
#     print("Episode finished after {} timesteps".format(t+1))
#     break





# self.get_num_tiles(): This function get the number of different tiles that can appear in the observation space
# get_border_tile(): This function get the tile index to be used for padding a certain problem. It is used by certain wrappers.
# adjust_param(**kwargs): This function that helps adjust the problem and/or representation parameters such as modifying width and height of the generated map.
