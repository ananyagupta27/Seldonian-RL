#!/usr/bin/env bash

set -exu
ENV=$1
EPISODES=$2

python main.py $ENV $EPISODES
