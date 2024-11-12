#! /bin/bash
PROCESSES=(
    "gz.*sim"
    "colored_line_following.sdf"
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


#------------ COMMON DEFINITIONS ----------------------
SRC_PATH=""
PROJECT_DIR="regelum-playground"
BUFFER_SIZE=20000
echo ARGS $#
if [ "$#" == "1" ] ; then
BUFFER_RESET=${1}
echo "BUFFER_RESET" ${BUFFER_RESET}
elif [ "$#" == "2" ]
then
BUFFER_RESET=${1}
BUFFER_SIZE=${2}
echo "BUFFER_RESET:" ${BUFFER_RESET} " BUFFER_RESET:" ${BUFFER_SIZE}
fi 
ROOT_PATH="${SRC_PATH}/${PROJECT_DIR}"
#-------------------------------------------------------


# PYTHONPATH - PYTHONPATh - PYTHONPATH --------------------------------
export PYTHONPATH=$PYTHONPATH:${ROOT_PATH}/src
export PYTHONPATH=$PYTHONPATH:${SRC_PATH}/sccl/src
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

#ps -ef | grep gz

# start gazebo
# sim_options=" -r -s --headless-rendering --render-engine ogre2 "
sim_options=" -r --render-engine ogre2"
gz sim ${sim_options} "${ROOT_PATH}/models/colored_line_following.sdf"  &

# debug
#ps -ef | grep gz

# start RL
echo  Executing Experiment

REHYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python3 run.py \
    scenario=sac_line_following \
    simulator=gz_3w_lf \
    system=3wrobot_line_following \
    running_objective=3wrobot_line_following \
    scenario.autotune=False \
    scenario.policy_lr=0.00079 \
    scenario.q_lr=0.00025 \
    scenario.alpha=0.0085 \
    scenario.learning_starts=250 \
    scenario.total_timesteps=5000 \
    scenario.checkpoint_dirpath="/regelum-playground/regelum_data/outputs/2024-11-12/13-17-48/0" \
    scenario.evaluation_only=true \
    +seed=42 \
    --experiment=sac_lf

echo DONE

# kill zombies
sleep 5s
# execute_watchout

#ps -ef | grep gz

