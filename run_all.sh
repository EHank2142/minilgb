#!/usr/bin/env bash
# Runs every experiment sequentially (timings must not overlap). ~3-4 h on a 6-core laptop.
set -e
cd "$(dirname "$0")"
export REPRO_THREADS=${REPRO_THREADS:-6}
python tests/test_correctness.py                 2>&1 | tee results/log_tests.txt
python experiments/exp_overall.py flight --iters 500 --skip-done 2>&1 | tee -a results/log_overall_flight.txt
python experiments/exp_overall.py mslr   --iters 500 --skip-done 2>&1 | tee -a results/log_overall_mslr.txt
python experiments/exp_table4.py                 2>&1 | tee results/log_table4.txt
python experiments/exp_efb.py                    2>&1 | tee results/log_efb.txt
python experiments/exp_goss_theory.py            2>&1 | tee results/log_goss_theory.txt
echo ALL DONE
