import logging
import gym

import matplotlib.pyplot as plt
import minerl

import coloredlogs
coloredlogs.install(logging.DEBUG)


env = gym.make("MineRLPunchCow-v0")
env.reset()

# ac = env.action_space.noop()
# # # ac["chat"] = "/give @p diamond 3"
# # ac["chat"] = "/summon minecraft:cow ^ ^ ^2"
# ac["camera"] = [10, 0.]  # look down a little bit to hit the cow
# env.step(ac)

damage_dealt = []
done = False

i = 0

while not done:
    ac = env.action_space.noop()
    i += 1
    ac["attack"] = i % 2  # punch instead of holding down the button
    # ac['camera'] = [0., 0.]
    obs, reward, done, info = env.step(ac)

    damage_dealt.append(obs["damage_dealt"]["damage_dealt"])

    env.render()
env.close()


# print(damage_dealt)
# print(damage_dealt[0])
plt.plot(damage_dealt)
plt.show()
