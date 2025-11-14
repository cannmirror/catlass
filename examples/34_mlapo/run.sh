n=1 #32
rmsNumCol2=2112


# 0:PER_TENSOR_ASYMM_QUANT 
# 1:PER_TOKEN_SYMM_QUANT 
quantMode=0

dtype=float16  # float16/bfloat16
device=5

function build() {
    rm -rf build
    rm -rf output
    bash ../../scripts/build.sh --enable_print 34_mlapo
}

# function gen_data() {
#     python3 gen_data.py $n $he $quantMode $dtype
#     echo "Data gen finished"
# }

function run_kernel {
    echo 'Case: n=' $n 
    cd ../../output/bin/
    ./34_mlapo $rmsNumCol2 $n $quantMode $dtype $device
    cd /home/chenyuning/1103catlass/catlass/examples/34_mlapo
}

build
# gen_data
run_kernel