#!/bin/bash

# Fall back to detecting from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."

# Upgrade pip
python3 -m pip install --upgrade pip --user

# Install required packages
python3 -m pip install --user -r "${WORKSPACE_DIR}/requirements.txt"

# Setup RAFT
export RAFT_DIR="${WORKSPACE_DIR}/src/demo-raft/notebooks/.gorilla"
sh "${WORKSPACE_DIR}/src/demo-raft/notebooks/setup_raft.sh"
