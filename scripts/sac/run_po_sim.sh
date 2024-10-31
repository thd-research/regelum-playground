#!/bin/bash
PROCESSES=(
    "gz.*sim"
    "colored_line_following.sdf"
    "models/pushing_objects.sdf"
    "gazebo_simulator"
    "ruby"
    "gz"
)



function print_message {
    echo ${3}
}

function print_info { print_message "BLUE"   "INFO" "${*}" ; }
function print_warn { print_message "YELLOW" "WARN" "${*}" ; }
function print_ok   { print_message "GREEN"  "OK"   "${*}" ; }
function print_err  { print_message "RED"    "ERR"  "${*}" ; }
function print_part { print_message "CYAN"   "PART" "${*}" ; }
function print_unk  { print_message "PURPLE" "UNK"  "${*}" ; }


function check_process {
    pgrep -f "${1}"
}



function eval_state {
    local state=$?

    if (( $state == 0 ))
        then print_ok "success ${1}"
        else print_err "failed ${1}"
    fi

    return $state
}


function kill_process {
    pkill -9 -f "${1}"
}


function execute_check {
    print_info "check process ${entry}"
    eval_state $(check_process "${entry}")
}

function execute_kill {
    print_info "try to kill ${entry}"
    eval_state $(kill_process "${entry}")
}

function execute_watchout {
    print_info "watchout for possible zombies"
    for entry in ${PROCESSES[@]}
    do
        execute_check &&
        execute_kill
    done
}

function execute_state {
    state=$?
    if (( $state == 0 ))
        then print_ok "success (${1})"
        else print_err "failed (${1})"
    fi
    return $state
}


# *------------ COMMON DEFINITIONS ----------------------
SRC_PATH=""
PROJECT_DIR="regelum-playground"
echo ARGS $#
if [ "$#" == "2" ] ; then
SRC_PATH=${1} ;
PROJECT_DIR=${2}
fi 
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
# *-------------------------------------------------------

# PYTHONPATH - PYTHONPATH - PYTHONPATH --------------------------------
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/src
# *--------------------------------------------------------------------

# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ
export GZ_VERSION="8"
export GZ_DISTRO="harmonic"
export GZ_IP="127.0.0.1"
export GZ_PARTITION="$(hostname)"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+${GZ_SIM_RESOURCE_PATH}:}${ROOT_PATH}/models"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ - GZ


# kill zombies
execute_watchout

# debug
#ps -ef | grep gz


# start gazebo
# sim_options=" -r -s --headless-rendering --render-engine ogre2"
sim_options=" -r --render-engine ogre2"
#sim_options=" -r --render-engine ogre"
gz sim ${sim_options} "${ROOT_PATH}/models/pushing_objects.sdf"  &

# debug
ps -ef | grep gz

# start RL
echo  Executing Experiment

REHYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python3 run.py \
    scenario=sac_pushing_object \
    simulator=gz_3w \
    system=3wrobot_pushing_object \
    running_objective=3wrobot_pushing_object \
    scenario.autotune=False \
    scenario.policy_lr=0.00079 \
    scenario.q_lr=0.00025 \
    scenario.alpha=0.0085 \
    scenario.total_timesteps=40 \
    +seed=4 \
    --fps=10

echo DONE

# kill zombies
execute_watchout

# debug
# ps -ef | grep gz

