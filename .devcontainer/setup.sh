#!/bin/bash

# Fall back to detecting from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."

# Setup RAFT
export RAFT_DIR="${WORKSPACE_DIR}/src/demo-raft/notebooks/.gorilla"
sh "${WORKSPACE_DIR}/src/demo-raft/notebooks/setup_raft.sh"
