import logging
import gym
import minerl
import coloredlogs

coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLFightZombie-v0")
env.reset()

done = False

while not done:
    # dont do anything
    ac = env.action_space.noop()
    obs, reward, done, info = env.step(ac)

    env.render()

env.close()
