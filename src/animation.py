from regelum.animation import ThreeWheeledRobotAnimation

import matplotlib.pyplot as plt

import omegaconf
from pathlib import Path


class ThreeWheeledRobotAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def setup(self):
        super().setup()

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
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
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        pass