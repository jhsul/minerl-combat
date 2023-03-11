<h1 align="center">
minerl-combat
</h1>
<p align="center">
This is a fork of <a href="https://github.com/minerllabs/minerl">minerl v1.0.0</a> with additional gym environments for player versus mob combat scenarios. Although combat is our focus, this package provides an interface for setting up environments via a list of minecraft chat commands; this can potentially be extended for many other scenarios.
</p>

## Requirements

These are not set in stone and you may be able to get this to work with different versions, but these are safe options that should work 🤞

- Linux (sorry Caleb) (Mac silicon tutorial coming soon)
- Java 8 ([Oracle recommended](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html))
- Python 3.9.16 (conda recommended)

## Installation

Remove any existing minerl installation

```sh
pip uninstall minerl -y
```

Clone this repository

```sh
git clone https://github.com/jhsul/minerl-combat
cd minerl-combat
```

Install with pip (this may take awhile)

```sh
pip install -r requirements.txt
pip install .
```

## Example Usage

This is a super basic example with no RL or even any real model. It just shows how you can set up the environment and control the agent. VPT integration coming soon.

```python
# tests/example.py
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

```
