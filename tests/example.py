import logging
import gym
import minerl
import coloredlogs

coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLPunchCow-v0")
env.reset()

done = False

while not done:
    # just go nuts
    ac = env.action_space.sample()
    obs, reward, done, info = env.step(ac)

    env.render()

env.close()
