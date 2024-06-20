from regelum.callback import ScenarioStepLogger
from src.scenario import ROSScenario

from typing import Dict, Any
import numpy as np


class ROSScenarioStepLogger(ScenarioStepLogger):
    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, ROSScenario)
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
                    f"time: {output['time']:.4f} ({100 * output['time']/obj.time_final:.1f}%), "
                    f"episode: {int(output['episode_id'])}/{obj.N_episodes}, "
                    f"iteration: {int(output['iteration_id'])}/{obj.N_iterations}"
                )
        except Exception as err:
            print(err)
            print("Error Here")