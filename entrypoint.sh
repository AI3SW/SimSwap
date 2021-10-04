#!/bin/bash
set -e

# for conda init
. /opt/conda/etc/profile.d/conda.sh
conda activate simswap
exec "$@"
