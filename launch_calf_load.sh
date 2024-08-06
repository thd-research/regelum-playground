# python3 run.py policy=3wrobot_MPC initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python3.10 run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1 --configure
ROOT="regelum_data/outputs"
CALF_MODEL_PATH=${ROOT} + "/2024-08-06/13-29-51/0/.callbacks/PolicyNumpyModelSaver/model_it_00028.npy"

# python3.10 run.py policy=rc_calfq \
#                   initial_conditions=3wrobot_kin_with_spot \
#                   scenario=my_scenario \
#                   system=3wrobot_kin_with_spot \
#                   common.sampling_time=0.1 \
#                   simulator.time_final=10 scenario.N_iterations=1 \
#                   policy.weight_path=${CALF_MODEL_PATH} \
#                   --interactive --fps=10

if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then  
        python3.10 run.py \
                  +seed=7 \
                  simulator=ros \
                  policy=rc_calfq \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.nominal_kappa_params="[0.1, 1.5, -.15]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=40 scenario.N_iterations=1 \
                  --jobs=-1 \
                  --experiment=calf_inc_penalty \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e3 \
                  policy.step_size_multiplier=5 \
                  policy.weight_path=${CALF_MODEL_PATH} \
                  policy.nominal_only=False \
                  simulator.use_phy_robot=True \
                  --interactive
    else
        python3.10 run.py +seed=7 \
                  policy=rc_calfq \
                  initial_conditions=3wrobot_kin_with_spot \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=40 scenario.N_iterations=40 \
                  --jobs=-1 \
                  --experiment=calf_inc_penalty \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e2 \
                  policy.step_size_multiplier=5 \
                  policy.nominal_only=False \
                  --interactive --fps=10
fi
