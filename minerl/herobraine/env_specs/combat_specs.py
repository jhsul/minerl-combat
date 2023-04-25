from typing import List, Optional, Sequence

from abc import abstractmethod

import gym

from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec


class TimeoutWrapper(gym.Wrapper):
    """Timeout wrapper specifically crafted for the BASALT environments"""

    def __init__(self, env):
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0  # THIS WAS WHY THE ENV COULDN'T RESET!
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class EndOnKillWrapper(gym.Wrapper):
    """
    This wrapper ends the episode when the agent kills a mob
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if obs["mob_kills"]["mob_kills"] > 0:
            done = True
        return obs, reward, done, info


class CalculateRewardsWrapper(gym.Wrapper):
    """
    This wrapper does the reward calculation for the combat environments
    """

    def __init__(self, env):
        super().__init__(env)
        self.stats = {
            "damage_dealt": [2, 0],
            "damage_taken": [-1, 0],
            "mob_kills": [100, 0]
        }
        self.time_punishment = -1

    def reset(self):
        for stat in self.stats:
            self.stats[stat][1] = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        rewards = []
        for k, v in self.stats.items():
            curr = obs[k][k]
            if curr > v[1]:
                rewards.append(v[0]*(curr - v[1]))
                v[1] = curr
                # print(f"{k} reward: {rewards[-1]}")

        reward = sum(rewards) + self.time_punishment

        return obs, reward, done, info


class InitCommandsWrapper(gym.Wrapper):
    """
    This wrapper injects minecraft chat commands into env.reset()
    It uses the init_cmds field of CombatBaseEnvSpec
    """

    def __init__(self, env, env_spec):
        super().__init__(env)
        self.env_spec = env_spec

    def run_command(self, cmd: str):
        ac = self.env.action_space.noop()
        ac["chat"] = cmd
        obs, reward, done, info = self.env.step(ac)
        return obs

    def reset(self):
        obs = super().reset()
        # print("Injecting minecraft chat commands into env.reset()!")

        for cmd in self.env_spec.init_cmds():
            # print(f"Running command: {cmd}")
            obs = self.run_command(cmd)

        return obs


def _combat_gym_entrypoint(
        env_spec: "CombatBaseEnvSpec",
        fake: bool = False,
) -> _singleagent._SingleAgentEnv:
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = TimeoutWrapper(env)
    env = InitCommandsWrapper(env, env_spec)
    env = EndOnKillWrapper(env)
    env = CalculateRewardsWrapper(env)
    return env


COMBAT_GYM_ENTRY_POINT = "minerl.herobraine.env_specs.combat_specs:_combat_gym_entrypoint"


class CombatBaseEnvSpec(HumanControlEnvSpec):

    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 1024

    # It would be cleaner to make this an instance variable
    # But it's just easier for InitCommandsWrapper
    # Since it gets the class itself and not an instance
    @staticmethod
    @abstractmethod
    def init_cmds() -> List[str]:
        pass

    def __init__(
            self,
            name,
            demo_server_experiment_name,
            max_episode_steps=2400,
            inventory: Sequence[dict] = (),
    ):
        # Used by minerl.util.docs to construct Sphinx docs.
        self.inventory = inventory
        self.demo_server_experiment_name = demo_server_experiment_name

        super().__init__(
            name=name,
            # This way, the setup actions are not counted as part of the episode.
            max_episode_steps=max_episode_steps + len(self.init_cmds()),
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=[640, 360],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0]
        )

    def is_from_folder(self, folder: str) -> bool:
        # Implements abstractmethod.
        return folder == self.demo_server_experiment_name

    def _entry_point(self, fake: bool) -> str:
        # Don't need to inspect `fake` argument here because it is also passed to the
        # entrypoint function.
        return COMBAT_GYM_ENTRY_POINT

    def create_observables(self):
        return [  # The POV in pixels
            handlers.POVObservation(self.resolution),
            # https://minecraft.fandom.com/wiki/Statistics#List_of_custom_statistic_names

            handlers.ObserveFromFullStats("damage_dealt"),
            handlers.ObserveFromFullStats("damage_taken"),
            handlers.ObserveFromFullStats("mob_kills"),

            # idk what this is
            handlers.ObservationFromLifeStats()]

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [
            handlers.SimpleInventoryAgentStart(self.inventory),
            handlers.DoneOnDeath()
        ]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> List[handlers.Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[handlers.Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (self.max_episode_steps * mc.MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> List[handlers.Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]

    def get_blacklist_reason(self, npz_data: dict) -> Optional[str]:
        """
        Some saved demonstrations are bogus -- they only contain lobby frames.

        We can automatically skip these by checking for whether any snowballs
        were thrown.
        """
        # TODO(shwang): Waterfall demos should also check for water_bucket use.
        #               AnimalPen demos should also check for fencepost or fence gate use.
        # TODO Clean up snowball stuff (not used anymore)
        equip = npz_data.get("observation$equipped_items$mainhand$type")
        use = npz_data.get("action$use")
        if equip is None:
            return f"Missing equip observation. Available keys: {list(npz_data.keys())}"
        if use is None:
            return f"Missing use action. Available keys: {list(npz_data.keys())}"

        assert len(equip) == len(use) + 1, (len(equip), len(use))

        for i in range(len(use)):
            if use[i] == 1 and equip[i] == "snowball":
                return None
        return "BasaltEnv never threw a snowball"

    def create_mission_handlers(self):
        # Implements abstractmethod
        return ()

    def create_monitors(self):
        # Implements abstractmethod
        return ()

    def create_rewardables(self):
        # Implements abstractmethod
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:
        """Implements abstractmethod.

        Basalt environment have no rewards, so this is always False."""
        return False

    def get_docstring(self):
        return self.__class__.__doc__


SECOND = 20
MINUTE = SECOND * 60


class PunchCowEzEnvSpec(CombatBaseEnvSpec):
    """
This is just the easy-mode cow punch environment. Same rules apply, except the cow has no AI and won't run away.
"""

    @staticmethod
    def init_cmds():
        return [
            # No distractions!
            "/kill @e[type=!player]",
            # Clear a platform
            "/setblock ^ ^ ^2 air",
            "/setblock ^ ^1 ^2 air",
            "/setblock ^ ^ ^1 air",
            "/setblock ^ ^1 ^1 air",
            # Spawn a cow 2 blocks in front of the player
            "/summon cow ^ ^ ^2 {NoAI:1,Health:10000}"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLPunchCowEz-v0",
            demo_server_experiment_name="punchcowez",
            max_episode_steps=10*SECOND,
            inventory=[],
        )


class PunchCowEzTestEnvSpec(CombatBaseEnvSpec):
    """
Same as PunchCowEz, except the cow can actually die
"""

    @staticmethod
    def init_cmds():
        return [
            # No distractions!
            "/kill @e[type=!player]",
            # Clear a platform
            "/setblock ^ ^ ^2 air",
            "/setblock ^ ^1 ^2 air",
            "/setblock ^ ^ ^1 air",
            "/setblock ^ ^1 ^1 air",
            # Spawn a cow 2 blocks in front of the player
            "/summon cow ^ ^ ^2 {NoAI:1}"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLPunchCowEzTest-v0",
            demo_server_experiment_name="punchcoweztest",
            max_episode_steps=10*SECOND,
            inventory=[],
        )


class PunchCowEnvSpec(CombatBaseEnvSpec):
    """
You spawn in a random world with a cow in front of you. You have 10 seconds to beat the crap out of the cow
"""

    @staticmethod
    def init_cmds():
        return [
            # No distractions!
            "/kill @e[type=!player]",
            # Clear a platform
            "/setblock ^ ^ ^2 air",
            "/setblock ^ ^1 ^2 air",
            "/setblock ^ ^ ^1 air",
            "/setblock ^ ^1 ^1 air",
            # Spawn a cow 2 blocks in front of the player
            "/summon cow ^ ^ ^2"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLPunchCow-v0",
            demo_server_experiment_name="punchcow",
            max_episode_steps=10*SECOND,
            inventory=[],
        )


class FightSkeletonEnvSpec(CombatBaseEnvSpec):
    """
Fight the skeleton for 10 seconds!
"""

    @staticmethod
    def init_cmds():
        return [
            "/time set midnight",
            "/kill @e[type=!player]",
            "/summon skeleton ^ ^ ^2",
            "/replaceitem entity @p weapon.offhand shield"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLFightSkeleton-v0",
            demo_server_experiment_name="fightskeleton",
            max_episode_steps=10*SECOND,
            inventory=[
                dict(type="diamond_sword", quantity=1),
            ],
        )


class FightZombieEnvSpec(CombatBaseEnvSpec):
    """
Fight the zombie for 10 seconds!
"""

    @staticmethod
    def init_cmds():
        return [
            "/time set midnight",
            "/kill @e[type=!player]",
            # "/difficuly peaceful",  # assume no other mobs spawn
            # Clear a platform
            "/setblock ^ ^ ^2 air",
            "/setblock ^ ^1 ^2 air",
            "/setblock ^ ^ ^1 air",
            "/setblock ^ ^1 ^1 air",
            # Spawn the zombie and turn it to face the player
            "/summon zombie ^ ^ ^2 {IsBaby:0,Rotation:[180.0f,0.0f]}",
            # "/execute as @e[type=zombie] at @s run tp @s ~ ~ ~ 180 0 true",
            # "/tp @e[type=zombie, dx=5, dy=5, dz=5] ^ ^ 2 facing ^ ^ ^"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLFightZombie-v0",
            demo_server_experiment_name="fightzombie",
            max_episode_steps=10*SECOND,
            inventory=[
                dict(type="diamond_sword", quantity=1),
            ],
        )


class EnderdragonEnvSpec(CombatBaseEnvSpec):
    """
You spawn in the end. Kill the enderdragon and beat the game!
"""

    @staticmethod
    def init_cmds():
        return [
            # This will send homie to the end
            "/setblock ~ ~ ~ minecraft:end_portal"
        ]

    def __init__(self):
        super().__init__(
            name="MineRLEnderdragon-v0",
            demo_server_experiment_name="enderdragon",
            max_episode_steps=5*MINUTE,
            inventory=[
                dict(type="diamond_sword", quantity=1),
                dict(type="bow", quantity=1),
                dict(type="arrow", quantity=64),
                dict(type="diamond_helmet", quantity=1),
                dict(type="diamond_chestplate", quantity=1),
                dict(type="diamond_leggings", quantity=1),
                dict(type="diamond_boots", quantity=1),
                dict(type="cobblestone", quantity=64),
                dict(type="steak", quantity=64),
            ],
        )
