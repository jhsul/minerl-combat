import logging
import gym
import minerl
import coloredlogs

coloredlogs.install(logging.DEBUG)

env = gym.make("MineRLPunchCow-v0")
env.reset()

ac = env.action_space.noop()

# look down 10 degrees to aim at the cow
ac["camera"] = [10., 0]

env.step(ac)

done = False

i = 0
while not done:
    ac = env.action_space.noop()

    # Alternate so we don't hold down the button
    ac["attack"] = i % 2
    obs, reward, done, info = env.step(ac)

    env.render()
    i += 1

env.close()
