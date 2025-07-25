#!/bin/bash

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

RAFT_DIR=${RAFT_DIR:-${SCRIPT_DIR}/.gorilla}
echo "Setup Gorilla RAFT in ${RAFT_DIR}"
if [ ! -d "${RAFT_DIR}" ]; then
    echo "Checking out the Gorilla RAFT repo"
    git clone --no-checkout --depth=1 --filter=tree:0 https://github.com/cedricvidal/gorilla.git --branch raft-distillation-recipe --single-branch ${RAFT_DIR}
fi
cd ${RAFT_DIR}
git config --global --add safe.directory ${RAFT_DIR}
git sparse-checkout set --no-cone raft
git checkout
git pull
