import os
import logging
import shutil

ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]

CATLASS_TEST_PATH = os.path.dirname(__file__)
CATLASS_TEST_TMP_PATH = os.path.join("/tmp", "catlass_test")
CATLASS_TEST_KERNEL_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "kernel")
CATLASS_TEST_INCLUDE_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "include")

CATLASS_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "catlass")
CATLASS_INCLUDE_PATH = CATLASS_TEST_INCLUDE_PATH

CATLASS_COMMIT_ID = ""

__LOG_LEVEL = logging._nameToLevel.get(
    os.environ.get("CATLASS_TEST_LOG_LEVEL", "INFO").upper(), logging.INFO
)

logging.basicConfig(
    level=__LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.makedirs(CATLASS_TEST_TMP_PATH, exist_ok=True)

if os.environ.get("CATLASS_TEST_CLEAN_KERNEL_CACHE"):
    shutil.rmtree(CATLASS_TEST_KERNEL_PATH, ignore_errors=True)
os.makedirs(CATLASS_TEST_KERNEL_PATH, exist_ok=True)

from catlass_test.interface.function import *
