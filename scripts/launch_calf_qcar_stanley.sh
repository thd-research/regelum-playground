python3.10 run.py \
            +seed=10 \
            simulator=ros_qcar \
            policy=rc_calf_qcar \
            scenario=calf_scenario \
            system=qcar_kin \
            common.sampling_time=0.1 \
            simulator.time_final=50 scenario.N_iterations=1 \
            --jobs=1 \
            --experiment=calf_qcar_load \
            policy.critic_desired_decay=1e-6 \
            policy.critic_low_kappa_coeff=1e-1 \
            policy.critic_up_kappa_coeff=8e2 \
            policy.penalty_factor=5e2 \
            policy.step_size_multiplier=1 \
            policy.nominal_only=true \
            --interactive
