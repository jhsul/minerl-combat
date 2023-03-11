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
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class DoneOnESCWrapper(gym.Wrapper):
    """
    Use the "ESC" action of the MineRL 1.0.0 to end
    an episode (if 1, step will return done=True)
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_over = False

    def reset(self):
        self.episode_over = False
        return self.env.reset()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError(
                "Expected `reset` after episode terminated, not `step`.")
        observation, reward, done, info = self.env.step(action)
        done = done or bool(action["ESC"])
        self.episode_over = done
        return observation, reward, done, info


class InitCommandsWrapper(gym.Wrapper):
    """
    This wrapper injects minecraft chat commands into env.reset()
    It uses the init_cmds field of FightMobBaseEnvSpec
    """

    def __init__(self, env, env_spec):
        super().__init__(env)
        self.env_spec = env_spec

    def run_command(self, cmd: str):
        ac = self.env.action_space.noop()
        ac["chat"] = cmd
        self.env.step(ac)

    def reset(self):
        super().reset()
        print("Injecting minecraft chat commands into env.reset()!")

        for cmd in self.env_spec.init_cmds():
            print(f"Running command: {cmd}")
            self.run_command(cmd)


def _fight_mob_gym_entrypoint(
        env_spec: "FightMobBaseEnvSpec",
        fake: bool = False,
) -> _singleagent._SingleAgentEnv:
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = TimeoutWrapper(env)
    env = DoneOnESCWrapper(env)
    env = InitCommandsWrapper(env, env_spec)
    return env


FIGHT_MOB_GYM_ENTRY_POINT = "minerl.herobraine.env_specs.fight_mob_specs:_fight_mob_gym_entrypoint"


class FightMobBaseEnvSpec(HumanControlEnvSpec):

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
            max_episode_steps=max_episode_steps,
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
        return FIGHT_MOB_GYM_ENTRY_POINT

    def create_observables(self):
        # Only POV
        return [handlers.POVObservation(self.resolution),
                handlers.ObserveFromFullStats("damage_dealt"),
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


class PunchCowEnvSpec(FightMobBaseEnvSpec):
    """
You spawn in a random world with a cow in front of you. You have 10 seconds to beat the shit out of the cow
"""

    @staticmethod
    def init_cmds():
        return [
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
