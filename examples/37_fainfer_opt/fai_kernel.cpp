/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/debug.hpp"

#include "kernel_common.hpp"

using namespace Catlass;

template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
    bool PAGED_CACHE_FLAG>
class FAInferKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    // Methods
    CATLASS_DEVICE
    FAInferKernel()
    {}

    struct TaskQue {
        uint32_t taskIdx;
        uint64_t gmOffsetV;
        uint64_t gmOffsetO;
        uint32_t rowNum;
        uint32_t stackSeqTile;
        uint32_t kvSIdx;
        uint32_t blockBOffset;
        uint32_t qSeqlen;
        uint32_t qSBlockSize;
        uint32_t qNBlockSize;
        uint32_t kvSLoopNumTotal;

        CATLASS_DEVICE
        TaskQue()
        {}

        CATLASS_DEVICE
        void SetValue(uint32_t taskIdx_, uint64_t gmOffsetV_, uint64_t gmOffsetO_, uint32_t rowNum_,
            uint32_t stackSeqTile_, uint32_t kvSIdx_, uint32_t blockBOffset_, uint32_t qSeqlen_, uint32_t qSBlockSize_,
            uint32_t qNBlockSize_, uint32_t kvSLoopNumTotal_)
        {
            taskIdx = taskIdx_;
            gmOffsetV = gmOffsetV_;
            gmOffsetO = gmOffsetO_;
            rowNum = rowNum_;
            stackSeqTile = stackSeqTile_;
            kvSIdx = kvSIdx_;
            blockBOffset = blockBOffset_;
            qSeqlen = qSeqlen_;
            qSBlockSize = qSBlockSize_;
            qNBlockSize = qNBlockSize_;
            kvSLoopNumTotal = kvSLoopNumTotal_;
        }
    };

    CATLASS_DEVICE void operator()(FAIKernelParams const &params)
    {
        __gm__ FATilingData *fATilingData = reinterpret_cast<__gm__ FATilingData *>(params.tiling);
        uint32_t batch = fATilingData->batch;
        uint32_t qHeads = fATilingData->numHeads;
        uint32_t kvHeads = fATilingData->kvHeads;
        uint32_t embed = fATilingData->embeddingSize;
        uint32_t pagedBlockSize = fATilingData->blockSize;
        uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
        uint32_t totalTaskNum = fATilingData->totalTaskNum;
        uint32_t maskType = fATilingData->maskType;
        float scaleValue = fATilingData->scaleValue;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTemp);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

#ifdef __DAV_C220_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ);
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

        uint64_t strideQO = qHeads * embed;
        uint64_t strideKV = kvHeads * embed;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
        uint32_t groupSize = qHeads / kvHeads;

        uint64_t qBOffset = 0;
        uint64_t kBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t oBOffset = 0;
        uint64_t blockBOffset = 0;

        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0; // 当前的seq
        uint32_t qSeqlen = static_cast<int64_t>(gActualQseqlen.GetValue(curBatch));
        uint32_t kvSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
        uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
        uint32_t curTotalTaskNum = firstBatchTaskNum;

        uint32_t blockStackNum = MAX_STACK_LEN / pagedBlockSize;
        uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
        // uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
        uint32_t loopCnt = 0;
        LayoutK layoutKTemp(strideKV, stackSeqTilePad);
        LayoutV layoutVTemp(stackSeqTilePad, strideKV);
        LayoutMask layOutMask(1024, 1024, 1024);

        TaskQue taskQue[PRE_LAUNCH + 1];

        uint32_t l1KPPingPongFlag = 0;
        uint32_t l0ABPingPongFlag = 0;
        uint32_t l0CPingPongFlag = 0;
        uint32_t workspaceStagesFlag = 0;
        // Go through each task.
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum + PRE_LAUNCH * coreNum; taskIdx += uint32_t(coreNum)) {
            // Get the offset of each core on the GM.
            while (taskIdx >= curTotalTaskNum && taskIdx < totalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                qBOffset += qSeqlen * strideQO;
                if constexpr (!PAGED_CACHE_FLAG) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                } else {
                    blockBOffset += maxNumBlocksPerBatch;
                }
                oBOffset += qSeqlen * strideQO;

                qSeqlen = reinterpret_cast<int64_t>(gActualQseqlen.GetValue(curBatch));
                kvSeqlen = reinterpret_cast<int64_t>(gActualKvseqlen.GetValue(curBatch));
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                curQSBlockTile = GetQSBlockTile(kvSeqlen);
                curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

            uint32_t kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qHeadIdx = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint64_t gmOffsetQ = qBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;
            uint64_t gmOffsetK = kBOffset + kvHeadIdx * embed;
            uint64_t gmOffsetV = vBOffset + kvHeadIdx * embed;
            uint64_t gmOffsetO = oBOffset + qSBlockIdx * curQSBlockTile * strideQO + qHeadIdx * embed;

            uint32_t qSBlockSize =
                (qSBlockIdx == (curQSBlockNum - 1)) ? (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1))
                                       ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                       : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            // uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

            uint32_t noSkipKvS = kvSeqlen;
            if (maskType != 0) {
                uint32_t diffS = kvSeqlen - qSeqlen;
                noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, noSkipKvS);
            }
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
            if (taskIdx >= totalTaskNum) {
                kvSLoopNumTotal = 1;
            }

            uint32_t stackSeqTile;
            // int32_t stackSeqCount = 0;

#ifdef __DAV_C220_CUBE__
            LayoutQ layoutQTemp(rowNum, embed);
            if (taskIdx < totalTaskNum) {
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
            }
#endif
            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal; kvSIdx += blockStackNum) {
                if (taskIdx < totalTaskNum) {
                    if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                    } else {
                        stackSeqTile = stackSeqTilePad;
                    }
                    // uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetS = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         workspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    uint32_t taskStagesFlag = (taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                    blockMmadQK(gQ[gmOffsetQ],
                        gK[gmOffsetK],
                        gS[gmOffsetS],
                        gBlockTable[blockBOffset],
                        layoutQTemp,
                        layoutKTemp,
                        layOutS,
                        actualBlockShapeQK,
                        kvSIdx,
                        pagedBlockSize,
                        strideKV,
                        l1KPPingPongFlag,
                        l0ABPingPongFlag,
                        l0CPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = gmOffsetS;

                    uint32_t triUp = noSkipKvS - qSBlockSize;
                    uint32_t triDown = noSkipKvS;
                    uint32_t kvSStartIdx = kvSIdx * pagedBlockSize;
                    uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                    bool isMask = triUp < kvSEndIdx;
                    if (isMask && maskType) {
                        epilogueOnlineSoftmax(gP[gmOffsetP],
                            gS[gmOffsetS],
                            gMask,
                            layOutP,
                            layOutS,
                            layOutMask,
                            actualBlockShapeQK,
                            (kvSIdx == 0),
                            qSBlockSize,
                            qNBlockSize,
                            workspaceStagesFlag,
                            taskStagesFlag,
                            qkReady,
                            triUp,
                            triDown,
                            kvSStartIdx,
                            kvSEndIdx);
                    } else {
                        Arch::CrossCoreWaitFlag(qkReady);
                        // online softmax
                        epilogueOnlineSoftmax(gP[gmOffsetP],
                            gS[gmOffsetS],
                            layOutP,
                            layOutS,
                            actualBlockShapeQK,
                            (kvSIdx == 0),
                            qSBlockSize,
                            qNBlockSize,
                            workspaceStagesFlag,
                            taskStagesFlag);
                    }
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    taskQue[workspaceStagesFlag].SetValue(taskIdx,
                        gmOffsetV,
                        gmOffsetO,
                        rowNum,
                        stackSeqTile,
                        kvSIdx,
                        blockBOffset,
                        qSeqlen,
                        qSBlockSize,
                        qNBlockSize,
                        kvSLoopNumTotal);
                }
                if (loopCnt >= PRE_LAUNCH) {
                    uint32_t nowWorkspaceStagesFlag =
                        (workspaceStagesFlag + (PRE_LAUNCH + 1) - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint32_t nowTaskStagesFlag = (taskQue[nowWorkspaceStagesFlag].taskIdx / coreNum) % (PRE_LAUNCH + 1);
                    uint32_t nowkvSIdx = taskQue[nowWorkspaceStagesFlag].kvSIdx;
                    uint32_t nowRowNum = taskQue[nowWorkspaceStagesFlag].rowNum;
                    stackSeqTile = taskQue[nowWorkspaceStagesFlag].stackSeqTile;
                    // uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetOTmp = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                            nowWorkspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{nowRowNum, embed, stackSeqTile};
                    LayoutOTmp layoutOTmp(nowRowNum, embed, embedRound);
#ifdef __DAV_C220_CUBE__
                    LayoutP layoutPTemp(nowRowNum, stackSeqTile, stackSeqTilePad);
                    uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                         nowWorkspaceStagesFlag * WORKSPACE_BLOCK_SIZE_DB;
                    blockMmadPV(gP[gmOffsetP],
                        gV[taskQue[nowWorkspaceStagesFlag].gmOffsetV],
                        gOTmp[gmOffsetOTmp],
                        gBlockTable[taskQue[nowWorkspaceStagesFlag].blockBOffset],
                        layoutPTemp,
                        layoutVTemp,
                        layoutOTmp,
                        actualBlockShapePV,
                        nowkvSIdx,
                        pagedBlockSize,
                        strideKV,
                        softmaxReady,
                        l1KPPingPongFlag,
                        l0ABPingPongFlag,
                        l0CPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutO layoutO(taskQue[nowWorkspaceStagesFlag].qSeqlen, embed * qHeads);
                    LayoutUpdate layoutUpdate(nowRowNum, embed, embedRound);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                    Arch::CrossCoreWaitFlag(pvReady);
                    // rescale O
                    epilogueRescaleO(gO[taskQue[nowWorkspaceStagesFlag].gmOffsetO],
                        gOTmp[gmOffsetOTmp],
                        gOUpdate[gmOffsetUpdate],
                        layoutO,
                        layoutOTmp,
                        layoutUpdate,
                        actualBlockShapePV,
                        taskQue[nowWorkspaceStagesFlag].qSBlockSize,
                        taskQue[nowWorkspaceStagesFlag].qNBlockSize,
                        (nowkvSIdx == 0),
                        nowkvSIdx + blockStackNum >= taskQue[nowWorkspaceStagesFlag].kvSLoopNumTotal,
                        nowWorkspaceStagesFlag,
                        nowTaskStagesFlag);
#endif
                }
                loopCnt++;
                workspaceStagesFlag = (workspaceStagesFlag + 1) % (PRE_LAUNCH + 1);
            }
        }
#ifdef __DAV_C220_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        // AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

template <class Dtype>
CATLASS_GLOBAL void FAInfer(uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR s, GM_ADDR p, GM_ADDR oTemp, GM_ADDR oUpdate,
    GM_ADDR tiling)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = Dtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = Dtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = Dtype;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = Dtype;
    using LayoutP = layout::RowMajor;
    using ElementO = Dtype;
    using LayoutO = layout::RowMajor;
    using ElementMask = Dtype;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    // L1TileShape::K must be qk embdding
    using L1TileShapeQK = GemmShape<Q_BLK, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    // GEMM Block模块，实现Flash Attention Infer的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKSplitRow<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxCasualMask;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    // L1TileShape::N must be v embdding
    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    // GEMM Block模块，实现Flash Attention Infer的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVSplitRow<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOSplitRow;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;

    // Kernel level
    // using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
    using FAInferKernel = FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, true>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, s, p, oTemp, oUpdate, tiling};

    // call kernel
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}
