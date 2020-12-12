#!/usr/bin/env bash

set -exu
ENV=$1
TRIAL=$2
python experiments.py --environment $ENV --trials $TRIAL
