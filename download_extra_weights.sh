#!/bin/bash

set -eu

PROJECT_ROOT=$(dirname $0)
VENV_DIR=$PROJECT_ROOT/venv_download

cd $PROJECT_ROOT

if [ $(ls $PROJECT_ROOT/models/Stable-diffusion | wc -l) -gt 1 ]; then
    echo "Too many checkpoints in $PROJECT_ROOT/models/Stable-diffusion"
    exit 1
fi

if [ -d $VENV_DIR ]; then
    rm -rf $VENV_DIR
fi

python -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install -r requirements_download.txt
python -c "from modules.initialize import initialize; initialize()"