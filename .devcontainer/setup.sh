CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup RAFT
export RAFT_DIR=${CUR_DIR}/../src/demo-raft/notebooks/.gorilla
sh ${CUR_DIR}/../src/demo-raft/notebooks/setup_raft.sh
