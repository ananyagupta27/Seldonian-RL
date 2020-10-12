
#!/usr/bin/env bash

set -exu

EPISODES=$1

python optimizers/cem.py $EPISODES