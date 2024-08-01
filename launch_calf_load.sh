# python3 run.py policy=3wrobot_MPC initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python3.10 run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1 --configure
ROOT="/home/tcc/huyhoang/regelum-playground/regelum_data/outputs"
CALF_MODEL_PATH=${ROOT} + "/2024-07-25/12-53-39/0/.callbacks/PolicyNumpyModelSaver/model_it_00007.npy"

python3.10 run.py policy=rc_calfq \
                  initial_conditions=3wrobot_kin_with_spot \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=10 scenario.N_iterations=1 \
                  policy.weight_path=${CALF_MODEL_PATH} \
                  --interactive --fps=10
