#!/bin/bash

# Fall back to detecting from script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="${SCRIPT_DIR}/.."

# Install required packages
pip install --user -r "${WORKSPACE_DIR}/requirements.txt"

# Setup RAFT
export RAFT_DIR="${WORKSPACE_DIR}/src/demo-raft/.gorilla"
sh "${WORKSPACE_DIR}/src/demo-raft/setup_raft.sh"

# Install markdown2pdf for converting markdown files to PDF
cargo install markdown2pdf
