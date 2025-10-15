CATLASS_REPO_URL=https://gitcode.com/cann/catlass.git
CATLASS_TAG=$1

git clone -b $CATLASS_TAG --depth=1 $CATLASS_REPO_URL /tmp/catlass_test_build
CATLASS_COMMIT_ID=$(git -C /tmp/catlass_test_build/ rev-parse HEAD)
CATLASS_COMMIT_ID_SHORT=$(git -C /tmp/catlass_test_build/ rev-parse --short HEAD) 

mv /tmp/catlass_test_build/include/catlass catlass_test/csrc/include/
mv /tmp/catlass_test_build/include/tla catlass_test/csrc/include/

sed -i "s/CATLASS_COMMIT_ID=\"\"/CATLASS_COMMIT_ID=\"${CATLASS_COMMIT_ID}\"/" catlass_test/__init__.py
sed -i "s/\(version = \"[0-9]\+\.[0-9]\+\.[0-9]\+\)\"/\1+${CATLASS_COMMIT_ID_SHORT}\"/" pyproject.toml
uv build 