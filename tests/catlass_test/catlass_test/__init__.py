import os
import logging

ASCEND_HOME_PATH = os.environ["ASCEND_HOME_PATH"]

CATLASS_TEST_PATH = os.path.dirname(__file__)
CATLASS_TEST_TMP_PATH = os.path.join("/tmp", "catlass_test")
CATLASS_TEST_KERNEL_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "kernel")
CATLASS_TEST_INCLUDE_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "include")
CATLASS_TEST_KERNEL_EXAMPLES_PATH = os.path.join(CATLASS_TEST_PATH, "csrc", "examples")

CATLASS_PATH = os.path.join(CATLASS_TEST_TMP_PATH, "catlass")
CATLASS_INCLUDE_PATH = CATLASS_TEST_INCLUDE_PATH

CATLASS_COMMIT_ID = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not os.path.exists(CATLASS_TEST_TMP_PATH):
    os.mkdir(CATLASS_TEST_TMP_PATH)
if not os.path.exists(CATLASS_TEST_KERNEL_PATH):
    os.mkdir(CATLASS_TEST_KERNEL_PATH)

from catlass_test.interface.function import *
