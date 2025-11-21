/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmHelper.h"

namespace ommhelper {
inline ID3D12Device5* OpacityMicroMapsHelper::GetD3D12Device5() {
    ID3D12Device* d3d12Device = (ID3D12Device*)NRI.GetDeviceNativeObject(m_Device);
    if (!d3d12Device) {
        printf("[FAILED] ID3D12Device* d3d12Device = NRI.GetDeviceNativeObject(*m_Device)");
        std::abort();
    }
    ID3D12Device5* d3d12Device5 = nullptr;
    if (d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12Device5)) != S_OK) {
        printf("[FAILED] d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12Device5))");
        std::abort();
    }
    return d3d12Device5;
}

inline ID3D12GraphicsCommandList4* OpacityMicroMapsHelper::GetD3D12GraphicsCommandList4(nri::CommandBuffer* commandBuffer) {
    ID3D12GraphicsCommandList4* commandList = nullptr;
    {
        ID3D12GraphicsCommandList* graphicsCommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(commandBuffer);
        if (graphicsCommandList->QueryInterface(IID_PPV_ARGS(&commandList)) != S_OK) {
            printf("[FAIL]: ID3D12GraphicsCommandList::QueryInterface(ID3D12GraphicsCommandList4)\n");
            std::abort();
        }
    }
    return commandList;
}

void OpacityMicroMapsHelper::InitializeD3D12() {
#if !DXR_OMM
    _NvAPI_Status nvResult = NvAPI_Initialize();
    if (nvResult != NVAPI_OK) {
        printf("[FAIL]: NvAPI_Initialize\n");
        std::abort();
    }

    _NVAPI_D3D12_SET_CREATE_PIPELINE_STATE_OPTIONS_PARAMS_V1 createPsoParams = {};
    createPsoParams.version = NVAPI_D3D12_SET_CREATE_PIPELINE_STATE_OPTIONS_PARAMS_VER;
    createPsoParams.flags = NVAPI_D3D12_PIPELINE_CREATION_STATE_FLAGS_ENABLE_OMM_SUPPORT;
    nvResult = NvAPI_D3D12_SetCreatePipelineStateOptions(GetD3D12Device5(), &createPsoParams);
    if (nvResult != NVAPI_OK) {
        printf("[FAIL]: NvAPI_D3D12_SetCreatePipelineStateOptions\n");
        std::abort();
    }
#endif
}

inline D3D12_RESOURCE_DESC InitBufferResourceDesc(size_t size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC result = {};
    result.Width = size;
    result.Flags = flags;
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    result.Height = 1;
    result.MipLevels = 1;
    result.DepthOrArraySize = 1;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc.Count = 1;
    result.SampleDesc.Quality = 0;
    result.Alignment = 0;

    return result;
}

inline D3D12_RESOURCE_BARRIER InitUavBarrier(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER result = {};
    result.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    result.UAV.pResource = resource;
    return result;
}

#if DXR_OMM

inline D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC FillOmmArrayDesc(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* ommArrayData, ID3D12Resource* ommDescArray) {
    uint64_t ommArrayDataOffset = inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].offset;
    uint64_t ommDescArrayOffset = inputs.buffers[(uint32_t)OmmDataLayout::DescArray].offset;

    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC desc;
    desc.NumOmmHistogramEntries = inputs.descArrayHistogramNum;
    desc.pOmmHistogram = (D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY*)inputs.descArrayHistogram;

    desc.InputBuffer = ommArrayData ? ommArrayData->GetGPUVirtualAddress() + ommArrayDataOffset : 128;              // has to be non-zero on prebuild
    desc.PerOmmDescs.StartAddress = ommDescArray ? ommDescArray->GetGPUVirtualAddress() + ommDescArrayOffset : 128; // has to be non-zero on prebuild
    desc.PerOmmDescs.StrideInBytes = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

    return desc;
}

inline D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS FillDefaultOmmArrayInputsDesc() {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS vmInput = {};
    vmInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    vmInput.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    vmInput.NumDescs = 1;
    vmInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

    return vmInput;
}

inline D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC FillGeometryTrianglesDesc(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* indexData, ID3D12Resource* vertexData) {
    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC trianglesDesc = {};
    trianglesDesc.IndexBuffer = indexData ? indexData->GetGPUVirtualAddress() + inputs.indices.offset : NULL;
    trianglesDesc.IndexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.indices.format);
    trianglesDesc.IndexCount = (UINT)inputs.indices.numElements;

    trianglesDesc.VertexCount = (UINT)inputs.vertices.numElements;
    trianglesDesc.VertexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.vertices.format);
    trianglesDesc.VertexBuffer.StrideInBytes = inputs.vertices.stride;
    trianglesDesc.VertexBuffer.StartAddress = vertexData ? vertexData->GetGPUVirtualAddress() + inputs.vertices.offset : NULL;

    return trianglesDesc;
}

inline D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC FillGeometryOmmLinkageDesc(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* ommArray, ID3D12Resource* ommIndexBuffer) {
    D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC odesc;
    odesc.OpacityMicromapArray = ommArray ? ommArray->GetGPUVirtualAddress() : NULL;
    odesc.OpacityMicromapBaseLocation = 0;
    odesc.OpacityMicromapIndexBuffer = {};

    size_t ommIndexOffset = inputs.buffers[(uint32_t)OmmDataLayout::Indices].offset;
    odesc.OpacityMicromapIndexBuffer.StartAddress = ommIndexBuffer ? ommIndexBuffer->GetGPUVirtualAddress() + ommIndexOffset : NULL;
    odesc.OpacityMicromapIndexBuffer.StrideInBytes = inputs.ommIndexStride;
    odesc.OpacityMicromapIndexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.ommIndexFormat);

    return odesc;
}

inline D3D12_RAYTRACING_GEOMETRY_DESC FillDefaultGeometryDesc() {
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
    geometryDesc.OmmTriangles = {};

    return geometryDesc;
}

inline D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS FillDefaultBlasInputsDesc() {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputDesc = {};
    inputDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputDesc.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    inputDesc.NumDescs = 1;
    inputDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    return inputDesc;
}

#else

inline NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS FillOmmArrayInputsDesc(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* ommArrayData, ID3D12Resource* ommDescArray) {
    NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = {};
    vmInput.flags = NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BUILD_FLAG_PREFER_FAST_TRACE;
    vmInput.numOMMUsageCounts = inputs.descArrayHistogramNum;
    vmInput.pOMMUsageCounts = (NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT*)inputs.descArrayHistogram;

    uint64_t ommArrayDataOffset = inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].offset;
    uint64_t ommDescArrayOffset = inputs.buffers[(uint32_t)OmmDataLayout::DescArray].offset;
    vmInput.inputBuffer = ommArrayData ? ommArrayData->GetGPUVirtualAddress() + ommArrayDataOffset : NULL;
    vmInput.perOMMDescs.StartAddress = ommDescArray ? ommDescArray->GetGPUVirtualAddress() + ommDescArrayOffset : NULL;
    vmInput.perOMMDescs.StrideInBytes = sizeof(_NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

    return vmInput;
}

inline NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX FillGeometryDescEx(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* indexData, ID3D12Resource* vertexData, ID3D12Resource* ommArray, ID3D12Resource* ommIndexBuffer) {
    NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = {};
    geometryDescEx.flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    geometryDescEx.type = NVAPI_D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES_EX;
    geometryDescEx.ommTriangles = {};

    NVAPI_D3D12_RAYTRACING_GEOMETRY_OMM_TRIANGLES_DESC& vmTriangles = geometryDescEx.ommTriangles;

    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& trianglesDesc = vmTriangles.triangles;
    trianglesDesc.IndexBuffer = indexData ? indexData->GetGPUVirtualAddress() + inputs.indices.offset : NULL;
    trianglesDesc.IndexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.indices.format);
    trianglesDesc.IndexCount = (UINT)inputs.indices.numElements;

    trianglesDesc.VertexCount = (UINT)inputs.vertices.numElements;
    trianglesDesc.VertexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.vertices.format);
    trianglesDesc.VertexBuffer.StrideInBytes = inputs.vertices.stride;
    trianglesDesc.VertexBuffer.StartAddress = vertexData ? vertexData->GetGPUVirtualAddress() + inputs.vertices.offset : NULL;

    vmTriangles.ommAttachment.opacityMicromapArray = ommArray ? ommArray->GetGPUVirtualAddress() : NULL;
    vmTriangles.ommAttachment.opacityMicromapBaseLocation = 0;
    vmTriangles.ommAttachment.opacityMicromapIndexBuffer = {};

    size_t ommIndexOffset = inputs.buffers[(uint32_t)OmmDataLayout::Indices].offset;
    vmTriangles.ommAttachment.opacityMicromapIndexBuffer.StartAddress = ommIndexBuffer ? ommIndexBuffer->GetGPUVirtualAddress() + ommIndexOffset : NULL;
    vmTriangles.ommAttachment.opacityMicromapIndexBuffer.StrideInBytes = inputs.ommIndexStride;
    vmTriangles.ommAttachment.opacityMicromapIndexFormat = (DXGI_FORMAT)nri::nriConvertNRIFormatToDXGI(inputs.ommIndexFormat);

    vmTriangles.ommAttachment.numOMMUsageCounts = inputs.indexHistogramNum;
    vmTriangles.ommAttachment.pOMMUsageCounts = (NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT*)inputs.indexHistogram;
    return geometryDescEx;
}

inline NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX FillDefaultBlasInputsDesc() {
    NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = {};
    inputDescEx.type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputDescEx.flags = NVAPI_D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE_EX;
    inputDescEx.numDescs = 1;
    inputDescEx.descsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputDescEx.geometryDescStrideInBytes = sizeof(NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX);
    return inputDescEx;
}

#endif

inline static size_t Align(size_t size, size_t alignment) {
    const size_t a = alignment;
    return ((size + a - 1) / a) * a;
}

void OpacityMicroMapsHelper::ReleaseMemoryD3D12() {
    if (m_D3D12ScratchBuffer)
        m_D3D12ScratchBuffer->Release();
    m_D3D12ScratchBuffer = nullptr;

    for (auto& heap : m_D3D12GeometryHeaps)
        heap->Release();
    m_D3D12GeometryHeaps.clear();

    m_CurrentHeapOffset = 0;
}

void OpacityMicroMapsHelper::AllocateMemoryD3D12(uint64_t size) {
    m_D3D12GeometryHeaps.reserve(16);
    ID3D12Device5* device = GetD3D12Device5();
    ID3D12Heap*& newHeap = m_D3D12GeometryHeaps.emplace_back();
    D3D12_HEAP_DESC desc = {};
    desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.SizeInBytes = size > m_DefaultHeapSize ? size : m_DefaultHeapSize;
    desc.SizeInBytes = m_D3D12ScratchBuffer ? desc.SizeInBytes : desc.SizeInBytes + m_SctrachSize;
    device->CreateHeap(&desc, IID_PPV_ARGS(&newHeap));
    m_CurrentHeapOffset = 0;

    if (!m_D3D12ScratchBuffer) {
        D3D12_RESOURCE_DESC resourceDesc = InitBufferResourceDesc(m_SctrachSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        GetD3D12Device5()->CreatePlacedResource(m_D3D12GeometryHeaps.back(), 0, &resourceDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_D3D12ScratchBuffer));
        m_CurrentHeapOffset += Align(m_SctrachSize, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    }
}

void OpacityMicroMapsHelper::BindResourceToMemoryD3D12(ID3D12Resource*& resource, size_t size) {
    if (m_D3D12GeometryHeaps.empty() || (m_CurrentHeapOffset + size) > m_DefaultHeapSize)
        AllocateMemoryD3D12(size);

    ID3D12Heap* heap = m_D3D12GeometryHeaps.back();

#if DXR_OMM
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
#else
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
#endif

    D3D12_RESOURCE_DESC resourceDesc = InitBufferResourceDesc(size, resourceFlags);
    GetD3D12Device5()->CreatePlacedResource(heap, m_CurrentHeapOffset, &resourceDesc, initialState, nullptr, IID_PPV_ARGS(&resource));
    m_CurrentHeapOffset += Align(size, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
}

void OpacityMicroMapsHelper::GetPreBuildInfoD3D12(MaskedGeometryBuildDesc** queue, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        MaskedGeometryBuildDesc& desc = *queue[i];
        { // get omm prebuild info
#if DXR_OMM
            {
                D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommDesc = FillOmmArrayDesc(desc.inputs, NULL, NULL);
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS vmInput = FillDefaultOmmArrayInputsDesc();
                vmInput.pOpacityMicromapArrayDesc = &ommDesc;

                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO ommPrebuildInfo = {};
                // Known issue in DXR 1.2: getting debug layer alignment error for an input buffer, which doesn't even get dereferenced on prebuild call.
                GetD3D12Device5()->GetRaytracingAccelerationStructurePrebuildInfo(&vmInput, &ommPrebuildInfo);
                desc.prebuildInfo.ommArraySize = ommPrebuildInfo.ResultDataMaxSizeInBytes;
                desc.prebuildInfo.maxScratchDataSize = ommPrebuildInfo.ScratchDataSizeInBytes;
            }
#else
            {
                NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = FillOmmArrayInputsDesc(desc.inputs, NULL, NULL);
                NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO ommPrebuildInfo = {};
                NVAPI_GET_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS ommGetPrebuildInfoParams = {};
                ommGetPrebuildInfoParams.pDesc = &vmInput;
                ommGetPrebuildInfoParams.pInfo = &ommPrebuildInfo;
                ommGetPrebuildInfoParams.version = NVAPI_GET_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS_VER;
                _NvAPI_Status nvResult = NvAPI_D3D12_GetRaytracingOpacityMicromapArrayPrebuildInfo(GetD3D12Device5(), &ommGetPrebuildInfoParams);
                if (nvResult != NVAPI_OK) {
                    printf("[FAIL]: NvAPI_D3D12_GetRaytracingOpacityMicromapArrayPrebuildInfo\n");
                    std::abort();
                }
                desc.prebuildInfo.ommArraySize = ommPrebuildInfo.resultDataMaxSizeInBytes;
                desc.prebuildInfo.maxScratchDataSize = ommPrebuildInfo.scratchDataSizeInBytes;
            }
#endif
        }

        { // get blas prebuild info
            nri::Buffer* nriOmmIndexData = desc.inputs.buffers[(uint32_t)OmmDataLayout::Indices].buffer;
            ID3D12Resource* ommIndexData = nriOmmIndexData ? (ID3D12Resource*)NRI.GetBufferNativeObject(nriOmmIndexData) : nullptr;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildInfo = {};
#if DXR_OMM
            {
                D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = FillDefaultGeometryDesc();
                D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC trianglesDesc = FillGeometryTrianglesDesc(desc.inputs, NULL, NULL);
                D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC ommDesc = FillGeometryOmmLinkageDesc(desc.inputs, NULL, ommIndexData);
                geometryDesc.OmmTriangles.pTriangles = &trianglesDesc;
                geometryDesc.OmmTriangles.pOmmLinkage = &ommDesc;

                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputDesc = FillDefaultBlasInputsDesc();
                inputDesc.pGeometryDescs = &geometryDesc;

                GetD3D12Device5()->GetRaytracingAccelerationStructurePrebuildInfo(&inputDesc, &blasPrebuildInfo);
            }
#else
            {
                NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = FillGeometryDescEx(desc.inputs, NULL, NULL, NULL, ommIndexData);

                NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = FillDefaultBlasInputsDesc();
                inputDescEx.pGeometryDescs = &geometryDescEx;

                NVAPI_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_EX_PARAMS blaskGetPrebuildInfoParams = {};
                NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX asInputs = {};
                blaskGetPrebuildInfoParams.pInfo = &blasPrebuildInfo;
                blaskGetPrebuildInfoParams.pDesc = &inputDescEx;
                blaskGetPrebuildInfoParams.version = NVAPI_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_EX_PARAMS_VER;

                NvAPI_Status nvapiStatus = NVAPI_OK;
                nvapiStatus = NvAPI_D3D12_GetRaytracingAccelerationStructurePrebuildInfoEx(GetD3D12Device5(), &blaskGetPrebuildInfoParams);
                if (nvapiStatus != NVAPI_OK) {
                    printf("[FAIL]: NvAPI_D3D12_GetRaytracingAccelerationStructurePrebuildInfoEx\n");
                    std::abort();
                }
            }
#endif
            desc.prebuildInfo.blasSize = blasPrebuildInfo.ResultDataMaxSizeInBytes;
            desc.prebuildInfo.maxScratchDataSize = std::max(blasPrebuildInfo.ScratchDataSizeInBytes, desc.prebuildInfo.maxScratchDataSize);
        }
    }
}

void OpacityMicroMapsHelper::BuildOmmArrayD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer) {
    if (!desc.inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].buffer)
        return;

    ID3D12Resource* ommArrayData = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].buffer);
    ID3D12Resource* ommDescArray = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.inputs.buffers[(uint32_t)OmmDataLayout::DescArray].buffer);

    ID3D12Resource* ommArrayBuffer = nullptr;
    BindResourceToMemoryD3D12(ommArrayBuffer, desc.prebuildInfo.ommArraySize);

#if DXR_OMM
    {
        D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommDesc = FillOmmArrayDesc(desc.inputs, ommArrayData, ommDescArray);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS vmInput = FillDefaultOmmArrayInputsDesc();
        vmInput.pOpacityMicromapArrayDesc = &ommDesc;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC vmArrayDesc = {};
        vmArrayDesc.DestAccelerationStructureData = ommArrayBuffer->GetGPUVirtualAddress();
        vmArrayDesc.Inputs = vmInput;
        vmArrayDesc.ScratchAccelerationStructureData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC info = {};
        GetD3D12GraphicsCommandList4(commandBuffer)->BuildRaytracingAccelerationStructure(&vmArrayDesc, 0, &info);
    }
#else
    {
        NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = FillOmmArrayInputsDesc(desc.inputs, ommArrayData, ommDescArray);

        NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC vmArrayDesc = {};
        vmArrayDesc.destOpacityMicromapArrayData = ommArrayBuffer->GetGPUVirtualAddress();
        vmArrayDesc.inputs = vmInput;
        vmArrayDesc.scratchOpacityMicromapArrayData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

        NVAPI_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_PARAMS buildVmParams = {};
        buildVmParams.numPostbuildInfoDescs = 0;
        buildVmParams.pPostbuildInfoDescs = nullptr;
        buildVmParams.pDesc = &vmArrayDesc;
        buildVmParams.version = NVAPI_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_PARAMS_VER;

        NvAPI_Status nvapiStatus = NvAPI_D3D12_BuildRaytracingOpacityMicromapArray(GetD3D12GraphicsCommandList4(commandBuffer), &buildVmParams);
        if (nvapiStatus != NVAPI_OK) {
            printf("[FAIL]: NvAPI_D3D12_BuildRaytracingOpacityMicromapArray\n");
            std::abort();
        }
    }
#endif

    D3D12_RESOURCE_BARRIER barriers[] = {InitUavBarrier(m_D3D12ScratchBuffer)};
    GetD3D12GraphicsCommandList4(commandBuffer)->ResourceBarrier(_countof(barriers), barriers);

    nri::BufferD3D12Desc wrappedBufferDesc = {ommArrayBuffer, 0};
    NRI.CreateBufferD3D12(*m_Device, wrappedBufferDesc, desc.outputs.ommArray);
    ommArrayBuffer->Release(); // dereference the resource to ensure it's destruction via NRI
}

void OpacityMicroMapsHelper::BuildBlasD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer) {
    if (!desc.outputs.ommArray)
        return;

    ID3D12Resource* indexData = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.inputs.indices.nriBufferOrPtr.buffer);
    ID3D12Resource* vertexData = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.inputs.vertices.nriBufferOrPtr.buffer);
    ID3D12Resource* ommArray = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.outputs.ommArray);
    ID3D12Resource* ommIndexData = (ID3D12Resource*)NRI.GetBufferNativeObject(desc.inputs.buffers[(uint32_t)OmmDataLayout::Indices].buffer);

    ID3D12Resource* blas = nullptr;
    BindResourceToMemoryD3D12(blas, desc.prebuildInfo.blasSize);

#if DXR_OMM
    {
        D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = FillDefaultGeometryDesc();
        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC trianglesDesc = FillGeometryTrianglesDesc(desc.inputs, indexData, vertexData);
        D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC ommDesc = FillGeometryOmmLinkageDesc(desc.inputs, ommArray, ommIndexData);
        geometryDesc.OmmTriangles.pTriangles = &trianglesDesc;
        geometryDesc.OmmTriangles.pOmmLinkage = &ommDesc;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputDesc = FillDefaultBlasInputsDesc();
        inputDesc.pGeometryDescs = &geometryDesc;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC vmArrayDesc = {};
        vmArrayDesc.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        vmArrayDesc.Inputs = inputDesc;
        vmArrayDesc.ScratchAccelerationStructureData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC info = {};
        GetD3D12GraphicsCommandList4(commandBuffer)->BuildRaytracingAccelerationStructure(&vmArrayDesc, 0, &info);
    }
#else
    {
        NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = FillGeometryDescEx(desc.inputs, indexData, vertexData, ommArray, ommIndexData);
        NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = FillDefaultBlasInputsDesc();
        inputDescEx.pGeometryDescs = &geometryDescEx;

        NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_EX asDesc = {};
        asDesc.destAccelerationStructureData = blas->GetGPUVirtualAddress();
        asDesc.inputs = inputDescEx;
        asDesc.scratchAccelerationStructureData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

        NVAPI_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_EX_PARAMS asExParams = {};
        asExParams.numPostbuildInfoDescs = 0;
        asExParams.pPostbuildInfoDescs = nullptr;
        asExParams.pDesc = &asDesc;
        asExParams.version = NVAPI_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_EX_PARAMS_VER;

        NvAPI_Status nvapiStatus = NvAPI_D3D12_BuildRaytracingAccelerationStructureEx(GetD3D12GraphicsCommandList4(commandBuffer), &asExParams);
        if (nvapiStatus != NVAPI_OK) {
            printf("[FAIL]: NvAPI_D3D12_BuildRaytracingAccelerationStructureEx\n");
            std::abort();
        }
    }
#endif
    D3D12_RESOURCE_BARRIER barriers[] = {InitUavBarrier(m_D3D12ScratchBuffer)};
    GetD3D12GraphicsCommandList4(commandBuffer)->ResourceBarrier(_countof(barriers), barriers);

    nri::AccelerationStructureD3D12Desc asDesc = {};
    asDesc.d3d12Resource = blas;
    asDesc.buildScratchSize = desc.prebuildInfo.maxScratchDataSize;
    asDesc.updateScratchSize = desc.prebuildInfo.maxScratchDataSize;
    NRI.CreateAccelerationStructureD3D12(*m_Device, asDesc, desc.outputs.blas);
    blas->Release(); // dereference the resource to ensure it's destruction via NRI
}

void OpacityMicroMapsHelper::BuildMaskedGeometryD3D12(MaskedGeometryBuildDesc** queue, const size_t count, nri::CommandBuffer* commandBuffer) {
    GetPreBuildInfoD3D12(queue, count);

    for (size_t i = 0; i < count; ++i) { // build omm then blas to increase memory locality
        BuildOmmArrayD3D12(*queue[i], commandBuffer);
        BuildBlasD3D12(*queue[i], commandBuffer);
    }
}
} // namespace ommhelper