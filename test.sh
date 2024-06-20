python3.10 run.py \
            simulator=ros \
            initial_conditions=3wrobot_kin_customized \
            system=3wrobot_kin \
            scenario=mpc_scenario \
            scenario.running_objective.spot_gain=0 \
            scenario.prediction_horizon=3 \
            --interactive \
            --fps=10