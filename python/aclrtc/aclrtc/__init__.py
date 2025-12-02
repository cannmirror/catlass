import os
import ctypes
from typing import List, Optional
from enum import Enum

ASCEND_HOME_PATH = os.getenv(
    "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
)
aclError = int

# Prog 类型定义
Prog = ctypes.c_void_p


class Error(Enum):
    ACL_SUCCESS = 0  # 执行成功。
    ACL_ERROR_RTC_INVALID_PROG = 176000  # 无效的Prog (handle)。
    ACL_ERROR_RTC_INVALID_INPUT = 176001  # 除prog入参以外的入参错误。
    ACL_ERROR_RTC_INVALID_OPTION = 176002  # 编译选项错误。
    ACL_ERROR_RTC_COMPILATION = 176003  # 编译报错。
    ACL_ERROR_RTC_LINKING = 176004  # 链接报错。
    ACL_ERROR_RTC_NO_NAME_EXPR_AFTER_COMPILATION = 176005  # 编译后没有函数名。
    ACL_ERROR_RTC_NO_LOWERED_NAMES_BEFORE_COMPILATION = (
        176006  # 编译后核函数名无法转换成Mangling名称。
    )
    ACL_ERROR_RTC_NAME_EXPR_NOT_VALID = 176007  # 传入无效的核函数名。
    ACL_ERROR_RTC_CREATE_PROG_FAILED = 276000  # 创建Prog (handle) 失败。
    ACL_ERROR_RTC_OUT_OF_MEMORY = 276001  # 内存不足。
    ACL_ERROR_RTC_FAILURE = 576000  # RTC内部错误。


def str_list_to_c_char_p_pointer(str_list: List[str]) -> ctypes.POINTER(
    ctypes.c_char_p
):
    # Converts a list of Python strings to a ctypes array of c_char_p,
    # and then returns a POINTER(c_char_p) pointing to the first element.
    # This is often needed when passing a string list to a C API.
    if not str_list:
        return ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_char_p))
    c_array = (ctypes.c_char_p * len(str_list))()
    for i, s in enumerate(str_list):
        # ensure each string is converted to bytes (utf-8 encoding)
        c_array[i] = s.encode("utf-8")
    return ctypes.cast(c_array, ctypes.POINTER(ctypes.c_char_p))




def _load_aclrtc_lib():
    lib_path = os.path.join(ASCEND_HOME_PATH, "lib64/libacl_rtc.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"ACLRTC library not found at {lib_path}")
    return ctypes.CDLL(lib_path)


def _get_bin_data_size(prog: Prog) -> int:
    """aclError aclrtcGetBinDataSize(Prog prog, int *binDataSize)
    
    内部函数，用于获取二进制数据大小。
    
    :param prog: Prog 句柄
    :type prog: Prog
    :return: 二进制数据大小
    :rtype: int
    """
    func_handle = _load_aclrtc_lib().aclrtcGetBinDataSize
    func_handle.argtypes = [Prog, ctypes.POINTER(ctypes.c_int)]
    func_handle.restype = aclError
    bin_data_size = ctypes.c_int()
    error = func_handle(prog, ctypes.byref(bin_data_size))
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcGetBinDataSize failed with error: {error}")
    return bin_data_size.value


def get_bin_data(prog: Prog) -> bytes:
    """aclError aclrtcGetBinData(Prog prog, char *binData)

    :param prog: Prog 句柄
    :type prog: Prog
    :return: 二进制数据
    :rtype: bytes
    """
    bin_data_size = _get_bin_data_size(prog)
    func_handle = _load_aclrtc_lib().aclrtcGetBinData
    func_handle.argtypes = [Prog, ctypes.POINTER(ctypes.c_char)]
    func_handle.restype = aclError
    bin_data = ctypes.create_string_buffer(bin_data_size)
    error = func_handle(prog, bin_data)
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcGetBinData failed with error: {error}")
    return bin_data.value


def _get_compile_log_size(prog: Prog) -> int:
    """aclError aclrtcGetCompileLogSize(Prog prog, int *compileLogSize)
    
    内部函数，用于获取编译日志大小。
    
    :param prog: Prog 句柄
    :type prog: Prog
    :return: 编译日志大小
    :rtype: int
    """
    func_handle = _load_aclrtc_lib().aclrtcGetCompileLogSize
    func_handle.argtypes = [Prog, ctypes.POINTER(ctypes.c_int)]
    func_handle.restype = aclError
    compile_log_size = ctypes.c_int()
    error = func_handle(prog, ctypes.byref(compile_log_size))
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcGetCompileLogSize failed with error: {error}")
    return compile_log_size.value


def get_compile_log(prog: Prog) -> str:
    """aclError aclrtcGetCompileLog(Prog prog, char *compileLog)
    
    :param prog: Prog 句柄
    :type prog: Prog
    :return: 编译日志
    :rtype: str
    """
    compile_log_size = _get_compile_log_size(prog)
    func_handle = _load_aclrtc_lib().aclrtcGetCompileLog
    func_handle.argtypes = [Prog, ctypes.POINTER(ctypes.c_char)]
    func_handle.restype = aclError
    compile_log = ctypes.create_string_buffer(compile_log_size)
    error = func_handle(prog, compile_log)
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcGetCompileLog failed with error: {error}")
    return compile_log.value.decode("utf-8")


def compile_prog(prog: Prog, options: Optional[List[str]] = None) -> aclError:
    """aclError aclrtcCompileProg(Prog prog, int numOptions, const char **options)

    :param prog: Prog 句柄
    :type prog: Prog
    :param options: 编译选项列表，如果为 None 或空列表则传递 NULL
    :type options: List[str], optional
    :return: 错误码
    :rtype: aclError
    """
    if options is None:
        options = []
    func_handle = _load_aclrtc_lib().aclrtcCompileProg
    func_handle.argtypes = [
        Prog,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    func_handle.restype = aclError
    options_ptr = str_list_to_c_char_p_pointer(options)
    error = func_handle(prog, len(options), options_ptr)
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcCompileProg failed with error: {error}")
    return error


def create_prog(
    src: str, name: str, headers: Optional[List[str]] = None, include_names: Optional[List[str]] = None
) -> Prog:
    """aclError aclrtcCreateProg(Prog *prog, const char *src, const char *name, int numHeaders, const char **headers, const char **includeNames)

    :param src: 源代码字符串
    :type src: str
    :param name: 程序名称
    :type name: str
    :param headers: 头文件列表，如果为 None 或空列表则传递 NULL
    :type headers: List[str], optional
    :param include_names: 包含名称列表，如果为 None 或空列表则传递 NULL
    :type include_names: List[str], optional
    :return: Prog 句柄
    :rtype: Prog
    """
    if headers is None:
        headers = []
    if include_names is None:
        include_names = []
    assert len(headers) == len(include_names), (
        "headers and include_names must have the same length"
    )
    func_handle = _load_aclrtc_lib().aclrtcCreateProg
    func_handle.argtypes = [
        ctypes.POINTER(Prog),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    func_handle.restype = aclError
    prog = Prog()
    headers_ptr = str_list_to_c_char_p_pointer(headers)
    include_names_ptr = str_list_to_c_char_p_pointer(include_names)
    error = func_handle(
        ctypes.byref(prog),
        src.encode("utf-8"),
        name.encode("utf-8"),
        len(headers),
        headers_ptr,
        include_names_ptr,
    )
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcCreateProg failed with error: {error}")
    return prog


def destroy_prog(prog: Prog) -> aclError:
    """aclError aclrtcDestroyProg(Prog prog)

    :param prog: Prog 句柄
    :type prog: Prog
    :return: 错误码
    :rtype: aclError
    """
    func_handle = _load_aclrtc_lib().aclrtcDestroyProg
    func_handle.argtypes = [Prog]
    func_handle.restype = aclError
    error = func_handle(prog)
    if error != Error.ACL_SUCCESS.value:
        raise RuntimeError(f"aclrtcDestroyProg failed with error: {error}")
    return error
