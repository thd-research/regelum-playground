python3.10 run.py \
            +seed=7 \
            simulator=ros_qcar \
            policy=rc_calf_qcar \
            scenario=my_scenario \
            system=qcar_kin \
            common.sampling_time=0.1 \
            simulator.time_final=40 scenario.N_iterations=40 \
            --jobs=-1 \
            --experiment=calf_inc_penalty \
            policy.critic_desired_decay=1e-5 \
            policy.critic_low_kappa_coeff=1e-1 \
            policy.critic_up_kappa_coeff=1e3 \
            policy.penalty_factor=1e3 \
            policy.step_size_multiplier=5 \
            policy.nominal_only=False