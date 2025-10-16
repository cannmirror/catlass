source /home/workspace/wyf/cann/ascend-toolkit/set_env.sh
bash ../../scripts/build.sh --clean 19_mla
if [ $? -eq 0 ]; then
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    python3 gen_data.py 1 1 128 16 16 128 half
    ../../output/bin/19_mla 1 1 128 16 16 128
fi
