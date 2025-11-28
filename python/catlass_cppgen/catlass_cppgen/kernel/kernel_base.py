class KernelBase:
    def codegen(self)->str:
        return "__global__ __aicore__"