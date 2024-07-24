from regelum.callback import ScenarioStepLogger, ObjectiveTracker
from regelum.policy import Policy
from src.scenario import RosMPC

from typing import Dict, Any
import numpy as np


class ROSScenarioStepLogger(ScenarioStepLogger):
    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, RosMPC)
            and method == "post_compute_action"
        )

    def on_function_call(self, obj, method: str, output: Dict[str, Any]):
        print("Enter Here")
        
        try:
            with np.printoptions(precision=2, suppress=True):
                self.log(
                    f"runn. objective: {output['running_objective']:.2f}, "
                    f"state est.: {output['estimated_state'][0]}, "
                    f"observation: {output['observation'][0]}, "
                    f"action: {output['action'][0]}, "
                    f"value: {output['current_value']:.4f}, "
                    f"time: {output['time']:.4f} ({100 * output['time']/obj.simulator.time_final:.1f}%), "
                    f"episode: {int(output['episode_id'])}/{obj.N_episodes}, "
                    f"iteration: {int(output['iteration_id'])}/{obj.N_iterations}"
                )
        except Exception as err:
            print(err)
            print("Error Here")

class MyObjectiveTracker(ObjectiveTracker):
    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, Policy)
            and method == "post_obj_run"
        )

    def is_done_collecting(self):
        return hasattr(self, "objective")

    def on_function_call(self, obj, method, output):
        # print("post_obj_run:", output["running_objective"], output["current_value"])
        self.running_objective = output["running_objective"]
        self.value = output["current_value"]
        self.objective = np.array([self.value, self.running_objective])
        self.objective_naming = ["Value", "Running objective"]


class CALFScenarioStepLogger(ScenarioStepLogger):
    def on_function_call(self, obj, method: str, output: Dict[str, Any]):
        try:
            with np.printoptions(precision=2, suppress=True):
                self.log(
                    f"runn. objective: {output['running_objective']:.2f}, "
                    f"state est.: {output['estimated_state'][0]}, "
                    f"observation: {output['observation'][0]}, "
                    f"action: {output['action'][0]}, "
                    f"value: {output['current_value']:.4f}, "
                    f"time: {output['time']:.4f} ({100 * output['time']/obj.simulator.time_final:.1f}%), "
                    f"episode: {int(output['episode_id'])}/{obj.N_episodes}, "
                    f"iteration: {int(output['iteration_id'])}/{obj.N_iterations}"
                )
        except Exception as err:
            print(err)
            print("Error Here")