from regelum.animation import ThreeWheeledRobotAnimation, ObjectiveComponentAnimation, DeferredComposedAnimation

import matplotlib.pyplot as plt

import omegaconf
from pathlib import Path
from .callback import MyObjectiveTracker
import numpy as np
from src.trajectory import TrajectoryGenerator


class MyObjectiveAnimation(DeferredComposedAnimation, MyObjectiveTracker):
    def __deferred_init__(self):
        objective_dimension = len(self.objective)
        self._animation_classes = []
        for i in range(objective_dimension):

            def objectiveComponent(*args, component=i, **kwargs):
                return ObjectiveComponentAnimation(*args, component=component, fontsize=min(10.0, 15.0/objective_dimension**0.5), **kwargs)

            self._animation_classes.append(objectiveComponent)
        super().__deferred_init__()

    def is_target_event(self, obj, method, output, triggers):
        return MyObjectiveTracker in triggers

    def on_trigger(self, _):
        self.__deferred_init__()
        super().on_launch()

class ThreeWheeledRobotAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def setup(self):
        super().setup()

    def lim(self, *args, **kwargs):
        # self.ax.set_xlim(-1.2, 0.2)
        # self.ax.set_ylim(-1.2, 0.2)
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        pass

class QcarAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def setup(self, init_file_name="qcar_kin.yaml", policy_file_name="qcar_kin_stanley.yaml"):
        super().setup()
        config_state_init = omegaconf.OmegaConf.load(
            Path(__file__).parent.parent / "presets" / "initial_conditions" / init_file_name
        )["state_init"]
        traj_type = omegaconf.OmegaConf.load(
            Path(__file__).parent.parent / "presets" / "policy" / policy_file_name
        )["trajectory_gen"]["trajectory_type"]

        initial_state = eval(config_state_init.strip(" =").replace("numpy", "np"))
        
        x_ref, y_ref, theta_ref = TrajectoryGenerator(initial_state, traj_type).trajectory

        self.x_lim = (np.min(x_ref)-1, np.max(x_ref)+1)
        self.y_lim = (np.min(y_ref)-1, np.max(y_ref)+1)
        self.ax.plot(x_ref, y_ref, "-", lw=1)

        for i, (x, y) in enumerate(zip(x_ref, y_ref)):
            if i >= len(theta_ref):
                break

            plt.arrow(x, y, np.cos(theta_ref[i]), np.sin(theta_ref[i]), width=0.01)

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(*self.x_lim)
        self.ax.set_ylim(*self.y_lim)
        pass

class ThreeWheeledRobotAnimationWithSpot(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x-, y-plane limits and gaussian cost spot."""

    def setup(self, config_file_name="mpc_scenario.yaml"):
        super().setup()
        config_running_objective = omegaconf.OmegaConf.load(
            Path(__file__).parent.parent / "presets" / "scenario" / config_file_name
        )["running_objective"]
        center = (
            config_running_objective["spot_x_center"],
            config_running_objective["spot_y_center"],
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                config_running_objective["spot_std"],
                color="r",
                alpha=0.66,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                2 * config_running_objective["spot_std"],
                color="r",
                alpha=0.29,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                3 * config_running_objective["spot_std"],
                color="r",
                alpha=0.15,
            )
        )

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)

class ThreeWheeledRobotAnimationWithSpotNewLims(ThreeWheeledRobotAnimationWithSpot):
    def setup(self):
        super().setup(config_file_name="mpc_scenario_customized.yaml")

    def lim(self, *args, **kwargs):
        # self.ax.set_xlim(-4, 4)
        # self.ax.set_ylim(-4, 4)
        self.ax.set_xlim(-1.2, 0.2)
        self.ax.set_ylim(-1.2, 0.2)
        pass