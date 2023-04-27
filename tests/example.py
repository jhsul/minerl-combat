import logging
import gym
import minerl
import coloredlogs

# coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLFightSkeleton-v0")
while True:
    env.reset()
    done = False
    while not done:
        action = env.action_space.noop()
        obs, reward, done, info = env.step(action)
        env.render()
