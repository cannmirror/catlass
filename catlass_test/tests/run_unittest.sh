export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/$(uname -m)-linux/devlib/libascend_hal.so:$LD_LIBRARY_PATH
python3 test_compiler.py