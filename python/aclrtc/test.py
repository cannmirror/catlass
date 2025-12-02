import aclrtc

prog = aclrtc.create_prog("#include <kernel_operator.h>\n__global__ __aicore__ void test(){{}}", "test", [], [])
try:
    aclrtc.compile_prog(prog, ["--npu-arch=dav-2201"])
except Exception as e:
    print(e)
print(aclrtc.get_compile_log(prog))
aclrtc.destroy_prog(prog)
