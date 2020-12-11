#!/usr/bin/env bash

set -exu
ENV=$1
EPISODES=$2
TRIAL=$3
python experiments.py $ENV $EPISODES $TRIAL
