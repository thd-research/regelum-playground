# python3 run.py policy=3wrobot_MPC initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python3.10 run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1
# python run.py policy=3wrobot_kin_nomial initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 --jobs=1 --configure

python3.10 run.py policy=rc_calfq initial_conditions=3wrobot_kin_with_spot system=3wrobot_kin_with_spot common.sampling_time=0.1 simulator.time_final=20 scenario.N_iterations=4 --interactive --fps=10 --jobs=1
