import subprocess
from sklearn.model_selection import ParameterGrid


def run_calf_hyperparameters():
    grid = {
        "critic_desired_decays": [1e-5, 5e-5, 1e-4, 1e-6],
        "critic_low_kappa_coeffs": [1e-1, 1e-2, 5e-1],
        "critic_up_kappa_coeffs": [1e3, 1e2, 5e2, 50]
    }
    
    for params in list(ParameterGrid(grid)):
        print(params)
        subprocess.run([
            "python3.10", "run.py",
            "+seed=range(10,20)",
            "policy=rc_calfq",
            "initial_conditions=3wrobot_kin_with_spot",
            "scenario=my_scenario",
            "system=3wrobot_kin_with_spot",
            "common.sampling_time=0.1",
            "simulator.time_final=20",
            "scenario.N_iterations=40",
            "--jobs=-1",
            "--experiment=calf_hyper_0108_1612",
            "policy.critic_desired_decay={}".format(params["critic_desired_decays"]),
            "policy.critic_low_kappa_coeff={}".format(params["critic_low_kappa_coeffs"]),
            "policy.critic_up_kappa_coeff={}".format(params["critic_up_kappa_coeffs"]),
            ])
        



if __name__ == "__main__":
    run_calf_hyperparameters()
