# 环境准备

> **说明**：请确认[基础依赖](../README.md#-软硬件配套说明)、[NPU驱动](基础依赖、NPU驱动和固件已安装。)和固件已安装。

1. **安装社区版CANN toolkit包**

根据您所使用的[昇腾产品](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)类别，请下载对应的CANN开发套件包`Ascend-cann-toolkit_{version}_linux-{arch}.run`，下载链接见[CANN toolkit](https://www.hiascend.com/zh/developer/download/community/result?module=cann)（有关CATLASS的版本支持情况详见[软件硬件配套说明](../README.md#软件硬件配套说明)）。

随后安装CANN开发套件包（详请参考[CANN安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit))。

```bash
# 确保安装包有可执行权限
chmod +x Ascend-cann-toolkit_{version}_linux-{arch}.run
# 安装CANN toolkit包
./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install_path}
```

- `{version}`: CANN包版本号。
- `{arch}`: 系统架构。
- `{install_path}`: 指定安装路径，默认为`/usr/local/Ascend`

2. **使能CANN 环境**

安装完成后，执行下述指令即完成CANN环境使能。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/set_env.sh
# 指定路径安装
# source ${install_path}/set_env.sh
```

3. **下载源码**

将CATLASS代码仓下载到本地。

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/catlass.git
```

请进一步参考[快速入门](./docs/quickstart.md#matmul算子开发)以开始第一个算子开发。