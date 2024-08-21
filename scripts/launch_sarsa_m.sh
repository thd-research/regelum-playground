if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then  
        python3.10 run.py \
                  +seed=7 \
                  simulator=ros \
                  policy=rc_sarsa_m \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.R1_diag="[1, 100, 1e-1, 0, 0]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=25 scenario.N_iterations=20 \
                  --jobs=-1 \
                  --experiment=sarsa_m_init_ros \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e3 \
                  policy.step_size_multiplier=5
    else
        python3.10 run.py +seed=7 \
                  policy=rc_sarsa_m \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.R1_diag="[10, 100, 1e-1, 0, 0]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=15 scenario.N_iterations=20 \
                  --jobs=-1 \
                  --experiment=sarsa_m_init \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e3 \
                  policy.step_size_multiplier=5 \
                  --interactive --fps=10
fi
