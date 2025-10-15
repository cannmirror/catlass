source /home/workspace/wyf/cann/ascend-toolkit/set_env.sh
bash ../../scripts/build.sh --clean 35_row_broadcast_reduction_evt
if [ $? -eq 0 ]; then
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    ../../output/bin/35_row_broadcast_reduction_evt 256 512 1024 0
    # 随机生成一些用例
    # for m in 64 128 192 256 320; do
    #   for n in 128 256 384 512; do
    #     for k in 256 512 768 1024; do
    #       ./35_row_broadcast_reduction_evt $m $n $k 0
    #     done
    #   done
    # done
fi
