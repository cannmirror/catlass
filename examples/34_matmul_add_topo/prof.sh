source /home/workspace/wyf/cann/ascend-toolkit/set_env.sh
bash ../../scripts/build.sh --clean 34_matmul_add_topo
if [ $? -eq 0 ]; then
    # export ASCEND_SLOG_PRINT_TO_STDOUT=1
    msprof op ../../output/bin/34_matmul_add_topo 256 512 1024 0
    # 随机生成一些用例
    # for m in 64 128 192 256 320; do
    #   for n in 128 256 384 512; do
    #     for k in 256 512 768 1024; do
    #       ./32_matmul_add_evt $m $n $k 0
    #     done
    #   done
    # done
fi
