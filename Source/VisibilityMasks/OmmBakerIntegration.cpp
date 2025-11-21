/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmBakerIntegration.h"

#define NRI_ABORT_ON_FAILURE(result) \
    if ((result) != nri::Result::SUCCESS) \
        exit(1);

void OmmBakerGpuIntegration::Initialize(nri::Device& device) {
    m_Device = &device;

    uint32_t nriResult = (uint32_t)nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
    nriResult |= (uint32_t)nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI);
    if (nriResult != (uint32_t)nri::Result::SUCCESS) {
        printf("[FAIL]: nri::nriGetInterface\n");
        std::abort();
    }

    ommBakerCreationDesc bakerCreationDesc = ommBakerCreationDescDefault();
    bakerCreationDesc.type = ommBakerType_GPU;
    ommResult ommResult = ommCreateBaker(&bakerCreationDesc, &m_GpuBaker);
    if (ommResult != ommResult_SUCCESS) {
        printf("[FAIL]: ommCreateBaker\n");
        std::abort();
    }

    ommGpuPipelineConfigDesc bakePipelineDesc = ommGpuPipelineConfigDescDefault();
    bakePipelineDesc.renderAPI = NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VK ? ommGpuRenderAPI_Vulkan : ommGpuRenderAPI_DX12;

    ommResult = ommGpuCreatePipeline(m_GpuBaker, &bakePipelineDesc, &m_Pipeline);
    if (ommResult != ommResult_SUCCESS) {
        printf("[FAIL]: ommGpuCreatePipeline\n");
        std::abort();
    }

    ommResult = ommGpuGetPipelineDesc(m_Pipeline, &m_PipelineInfo);
    if (ommResult != ommResult_SUCCESS) {
        printf("[FAIL]: ommGpuGetPipelineDesc\n");
        std::abort();
    }

    nri::Queue* commandQueue = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, commandQueue));
    {
        CreateStaticResources(commandQueue);
        CreateSamplers(m_PipelineInfo);
        CreateTextures(m_PipelineInfo->pipelineNum);
        CreatePipelines(m_PipelineInfo);
    }
}

ommTexCoordFormat GetOmmTexcoordFormat(nri::Format format) {
    switch (format) {
        case nri::Format::RG16_UNORM:
            return ommTexCoordFormat_UV16_UNORM;
        case nri::Format::RG16_SFLOAT:
            return ommTexCoordFormat_UV16_FLOAT;
        case nri::Format::RG32_SFLOAT:
            return ommTexCoordFormat_UV32_FLOAT;
        default:
            printf("[FAIL] Unsupported texCoord format\n");
            std::abort();
    }
}

ommIndexFormat GetOmmIndexFormat(nri::Format inFormat) {
    switch (inFormat) {
        case nri::Format::R8_UINT:
            return ommIndexFormat_UINT_8;
        case nri::Format::R16_UINT:
            return ommIndexFormat_UINT_16;
        case nri::Format::R32_UINT:
            return ommIndexFormat_UINT_32;
        default:
            printf("[FAIL] Unsupported index format\n");
            std::abort();
    }
}

nri::Format GetNriIndexFormat(ommIndexFormat inFormat) {
    switch (inFormat) {
        case ommIndexFormat_UINT_8:
            return nri::Format::R8_UINT;
        case ommIndexFormat_UINT_16:
            return nri::Format::R16_UINT;
        case ommIndexFormat_UINT_32:
            return nri::Format::R32_UINT;
        default:
            printf("[FAIL] Unsupported index format\n");
            std::abort();
    }
}

ommTextureFilterMode GetOmmFilterMode(nri::Filter mode) {
    switch (mode) {
        case nri::Filter::LINEAR:
            return ommTextureFilterMode_Linear;
        case nri::Filter::NEAREST:
            return ommTextureFilterMode_Nearest;
        default:
            printf("[FAIL] Invalid ommTextureFilterMode\n");
            std::abort();
    }
}

ommTextureAddressMode GetOmmAddressingMode(nri::AddressMode mode) {
    switch (mode) {
        case nri::AddressMode::REPEAT:
            return ommTextureAddressMode_Wrap;
        case nri::AddressMode::MIRRORED_REPEAT:
            return ommTextureAddressMode_Mirror;
        case nri::AddressMode::CLAMP_TO_EDGE:
            return ommTextureAddressMode_Clamp;
        case nri::AddressMode::CLAMP_TO_BORDER:
            return ommTextureAddressMode_Border;
        default:
            printf("[FAIL] Invalid ommTextureAddressMode\n");
            std::abort();
    }
}

nri::DescriptorType GetNriDescriptorType(ommGpuDescriptorType ommType) {
    switch (ommType) {
        case ommGpuDescriptorType_TextureRead:
            return nri::DescriptorType::TEXTURE;
        case ommGpuDescriptorType_BufferRead:
            return nri::DescriptorType::BUFFER;
        case ommGpuDescriptorType_RawBufferRead:
            return nri::DescriptorType::STRUCTURED_BUFFER;
        case ommGpuDescriptorType_RawBufferWrite:
            return nri::DescriptorType::STORAGE_STRUCTURED_BUFFER;
        default:
            printf("[FAIL] Invalid ommGpuDescriptorType");
            std::abort();
    }
}

nri::AddressMode GetNriAddressMode(ommTextureAddressMode mode) {
    switch (mode) {
        case ommTextureAddressMode_Wrap:
            return nri::AddressMode::REPEAT;
        case ommTextureAddressMode_Mirror:
            return nri::AddressMode::MIRRORED_REPEAT;
        case ommTextureAddressMode_Clamp:
            return nri::AddressMode::CLAMP_TO_EDGE;
        case ommTextureAddressMode_Border:
            return nri::AddressMode::CLAMP_TO_BORDER;
        case ommTextureAddressMode_MirrorOnce:
            return nri::AddressMode::MIRRORED_REPEAT;
        default:
            printf("[FAIL] Invalid ommTextureAddressMode\n");
            std::abort();
    }
}

nri::Filter GetNriFilterMode(ommTextureFilterMode mode) {
    switch (mode) {
        case ommTextureFilterMode_Linear:
            return nri::Filter::LINEAR;
        case ommTextureFilterMode_Nearest:
            return nri::Filter::NEAREST;
        default:
            printf("[FAIL] Invalid ommTextureFilterMode\n");
            std::abort();
    }
}

nri::AccessBits GetNriResourceState(ommGpuDescriptorType descriptorType) {
    switch (descriptorType) {
        case ommGpuDescriptorType_BufferRead:
            return nri::AccessBits::SHADER_RESOURCE;
        case ommGpuDescriptorType_RawBufferRead:
            return nri::AccessBits::SHADER_RESOURCE;
        case ommGpuDescriptorType_RawBufferWrite:
            return nri::AccessBits::SHADER_RESOURCE_STORAGE;
        case ommGpuDescriptorType_TextureRead:
            return nri::AccessBits::SHADER_RESOURCE;
        default:
            printf("[FAIL] Invalid ommGpuDescriptorType\n");
            std::abort();
    }
}

BufferResource& OmmBakerGpuIntegration::GetBuffer(const ommGpuResource& resource, uint32_t geometryId) {
    BakerInputs& inputs = m_GeometryQueue[geometryId].desc->inputs;
    BakerOutputs& outputs = m_GeometryQueue[geometryId].desc->outputs;
    switch (resource.type) {
        case ommGpuResourceType_IN_TEXCOORD_BUFFER:
            return inputs.inUvBuffer;
        case ommGpuResourceType_IN_INDEX_BUFFER:
            return inputs.inIndexBuffer;
        case ommGpuResourceType_IN_SUBDIVISION_LEVEL_BUFFER:
            return inputs.inSubdivisionLevelBuffer;
        case ommGpuResourceType_OUT_OMM_ARRAY_DATA:
            return outputs.outArrayData;
        case ommGpuResourceType_OUT_OMM_DESC_ARRAY:
            return outputs.outDescArray;
        case ommGpuResourceType_OUT_OMM_INDEX_BUFFER:
            return outputs.outIndexBuffer;
        case ommGpuResourceType_OUT_OMM_DESC_ARRAY_HISTOGRAM:
            return outputs.outArrayHistogram;
        case ommGpuResourceType_OUT_OMM_INDEX_HISTOGRAM:
            return outputs.outIndexHistogram;
        case ommGpuResourceType_OUT_POST_DISPATCH_INFO:
            return outputs.outPostBuildInfo;
        case ommGpuResourceType_TRANSIENT_POOL_BUFFER:
            return inputs.inTransientPool[resource.indexInPool];
        case ommGpuResourceType_STATIC_VERTEX_BUFFER:
            return m_StaticBuffers[(uint32_t)GpuStaticResources::VertexBuffer];
        case ommGpuResourceType_STATIC_INDEX_BUFFER:
            return m_StaticBuffers[(uint32_t)GpuStaticResources::IndexBuffer];
        default:
            std::abort();
    }
}

nri::BufferViewType GetNriBufferViewType(ommGpuDescriptorType type) {
    switch (type) {
        case ommGpuDescriptorType_BufferRead:
            return nri::BufferViewType::SHADER_RESOURCE;
        case ommGpuDescriptorType_RawBufferRead:
            return nri::BufferViewType::SHADER_RESOURCE;
        case ommGpuDescriptorType_RawBufferWrite:
            return nri::BufferViewType::SHADER_RESOURCE_STORAGE;
        case ommGpuDescriptorType_TextureRead:
        default:
            printf("[FAIL] Invalid BufferDescriptorType\n");
            std::abort();
    }
}

ommGpuBakeFlags GetBakeFlags(BakerBakeFlags flags) {
    static_assert((uint32_t)BakerBakeFlags::Invalid == (uint32_t)ommGpuBakeFlags_Invalid);
    static_assert((uint32_t)BakerBakeFlags::EnablePostBuildInfo == (uint32_t)ommGpuBakeFlags_EnablePostDispatchInfoStats);
    static_assert((uint32_t)BakerBakeFlags::DisableSpecialIndices == (uint32_t)ommGpuBakeFlags_DisableSpecialIndices);
    static_assert((uint32_t)BakerBakeFlags::DisableTexCoordDeduplication == (uint32_t)ommGpuBakeFlags_DisableTexCoordDeduplication);
    static_assert((uint32_t)BakerBakeFlags::EnableNsightDebugMode == (uint32_t)ommGpuBakeFlags_EnableNsightDebugMode);
    return (ommGpuBakeFlags)flags;
}

ommGpuScratchMemoryBudget GetScratchMemoryBudget(BakerScratchMemoryBudget budget) {
    static_assert((uint64_t)ommGpuScratchMemoryBudget_Undefined == (uint64_t)BakerScratchMemoryBudget::Undefined);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_4 == (uint64_t)BakerScratchMemoryBudget::MB_4);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_32 == (uint64_t)BakerScratchMemoryBudget::MB_32);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_64 == (uint64_t)BakerScratchMemoryBudget::MB_64);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_128 == (uint64_t)BakerScratchMemoryBudget::MB_128);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_256 == (uint64_t)BakerScratchMemoryBudget::MB_256);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_512 == (uint64_t)BakerScratchMemoryBudget::MB_512);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_1024 == (uint64_t)BakerScratchMemoryBudget::MB_1024);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_Default == (uint64_t)BakerScratchMemoryBudget::Default);
    return (ommGpuScratchMemoryBudget)budget;
}

void FillDescriptorRangeDescs(uint32_t count, const ommGpuDescriptorRangeDesc* ommDesc, nri::DescriptorRangeDesc* nriDesc) {
    for (uint32_t i = 0; i < count; ++i) {
        nriDesc[i].baseRegisterIndex = ommDesc[i].baseRegisterIndex;
        nriDesc[i].descriptorNum = ommDesc[i].descriptorNum;
        nriDesc[i].descriptorType = GetNriDescriptorType(ommDesc[i].descriptorType);
        nriDesc[i].flags = nri::DescriptorRangeBits::NONE;
        nriDesc[i].shaderStages = nri::StageBits::ALL;
    }
}

void OmmBakerGpuIntegration::CreateGraphicsPipeline(uint32_t pipelineId, const ommGpuPipelineInfoDesc* pipelineInfo) {
    const ommGpuGraphicsPipelineDesc& pipelineDesc = pipelineInfo->pipelines[pipelineId].graphics;
    static_assert(OMM_GRAPHICS_PIPELINE_DESC_VERSION == 3, "ommGpuGraphicsPipelineDesc has changed\n");

    std::vector<nri::DescriptorRangeDesc> descriptorRangeDescs(pipelineDesc.descriptorRangeNum + 2); // + static samplers + constant buffer
    FillDescriptorRangeDescs(pipelineDesc.descriptorRangeNum, pipelineDesc.descriptorRanges, descriptorRangeDescs.data());

    nri::DescriptorRangeDesc& staticSamplersRange = descriptorRangeDescs[pipelineDesc.descriptorRangeNum + 0];
    staticSamplersRange.baseRegisterIndex = 0;
    staticSamplersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamplersRange.descriptorType = nri::DescriptorType::SAMPLER;
    staticSamplersRange.flags = nri::DescriptorRangeBits::NONE;
    staticSamplersRange.shaderStages = nri::StageBits::ALL;

    nri::DescriptorRangeDesc& constantBufferRange = descriptorRangeDescs[pipelineDesc.descriptorRangeNum + 1];
    constantBufferRange.baseRegisterIndex = pipelineInfo->globalConstantBufferDesc.registerIndex;
    constantBufferRange.descriptorNum = 1;
    constantBufferRange.descriptorType = nri::DescriptorType::CONSTANT_BUFFER;
    constantBufferRange.flags = nri::DescriptorRangeBits::NONE;
    constantBufferRange.shaderStages = nri::StageBits::ALL;

    nri::DescriptorSetDesc descriptorSetDescs = {};
    descriptorSetDescs.rangeNum = (uint32_t)descriptorRangeDescs.size();
    descriptorSetDescs.ranges = descriptorRangeDescs.data();

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = &descriptorSetDescs;
    layoutDesc.descriptorSetNum = 1;
    layoutDesc.shaderStages = nri::StageBits::GRAPHICS_SHADERS;
    nri::RootConstantDesc pushConstantDesc = {};
    {
        pushConstantDesc.registerIndex = pipelineInfo->localConstantBufferDesc.registerIndex;
        pushConstantDesc.size = pipelineInfo->localConstantBufferDesc.maxDataSize;
        pushConstantDesc.shaderStages = nri::StageBits::ALL;
        layoutDesc.rootConstants = &pushConstantDesc;
        layoutDesc.rootConstantNum = 1;
    }
    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, layoutDesc, m_NriPipelineLayouts.emplace_back()));

    nri::GraphicsPipelineDesc nriPipelineDesc = {};
    nriPipelineDesc.pipelineLayout = m_NriPipelineLayouts.back();
    nriPipelineDesc.multisample = nullptr;

    nri::VertexInputDesc vertexInputDesc = {};
    nriPipelineDesc.vertexInput = &vertexInputDesc;

    nri::VertexStreamDesc vertexStreamDesc = {};
    {
        vertexStreamDesc.bindingSlot = 0;
        vertexStreamDesc.stepRate = nri::VertexStreamStepRate::PER_VERTEX;
    }
    vertexInputDesc.streams = &vertexStreamDesc;
    vertexInputDesc.streamNum = 1;

    ommGpuGraphicsPipelineInputElementDesc inputElementDesc = ommGpuGraphicsPipelineInputElementDescDefault();
    nri::VertexAttributeDesc vertextAttributes = {};
    {
        vertextAttributes.format = nri::Format::R32_UINT;
        vertextAttributes.d3d.semanticIndex = inputElementDesc.semanticIndex;
        vertextAttributes.d3d.semanticName = inputElementDesc.semanticName;
        vertextAttributes.vk.location = 0;
        vertextAttributes.streamIndex = 0;
    }
    vertexInputDesc.attributes = &vertextAttributes;
    vertexInputDesc.attributeNum = 1;

    nri::InputAssemblyDesc& inputAssemblyDesc = nriPipelineDesc.inputAssembly;
    {
        inputAssemblyDesc.topology = nri::Topology::TRIANGLE_LIST;
        inputAssemblyDesc.tessControlPointNum = 0;
        inputAssemblyDesc.primitiveRestart = nri::PrimitiveRestart::DISABLED;
    }

    nri::RasterizationDesc& rasterizationDesc = nriPipelineDesc.rasterization;
    {
        rasterizationDesc.depthBias.clamp;
        rasterizationDesc.depthBias.constant;
        rasterizationDesc.depthBias.slope;
        rasterizationDesc.frontCounterClockwise;
        rasterizationDesc.lineSmoothing;
        rasterizationDesc.shadingRate;
        rasterizationDesc.depthClamp;

        rasterizationDesc.fillMode = nri::FillMode::SOLID;
        rasterizationDesc.cullMode = nri::CullMode::NONE;
        rasterizationDesc.conservativeRaster = pipelineDesc.conservativeRasterization;
    }

    nri::OutputMergerDesc& outputMergerDesc = nriPipelineDesc.outputMerger;
    std::vector<nri::ColorAttachmentDesc> colorAttachments;
    {
        outputMergerDesc.colorNum = pipelineDesc.numRenderTargets;
        for (uint32_t i = 0; i < outputMergerDesc.colorNum; ++i) {
            nri::ColorAttachmentDesc colorAttachment = {};
            colorAttachment.blendEnabled = false;
            colorAttachment.format = m_DebugTexFormat;
            colorAttachment.colorWriteMask = nri::ColorWriteBits::RGBA;
            colorAttachments.push_back(colorAttachment);
        }
        outputMergerDesc.colors = colorAttachments.data();
        outputMergerDesc.depth.write = false;
    }

    m_ColorDescriptorPerPipeline[pipelineId] = outputMergerDesc.colorNum ? m_DebugTextureDescriptor : m_EmptyDescriptor;

    std::vector<nri::ShaderDesc> shaderStages;
    if (pipelineDesc.vertexShader.data) {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.vertexShader.data;
        desc.size = pipelineDesc.vertexShader.size;
        desc.entryPointName = pipelineDesc.vertexShaderEntryPointName;
        desc.stage = nri::StageBits::VERTEX_SHADER;
    }
    if (pipelineDesc.geometryShader.data) {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.geometryShader.data;
        desc.size = pipelineDesc.geometryShader.size;
        desc.entryPointName = pipelineDesc.geometryShaderEntryPointName;
        desc.stage = nri::StageBits::GEOMETRY_SHADER;
    }
    if (pipelineDesc.pixelShader.data) {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.pixelShader.data;
        desc.size = pipelineDesc.pixelShader.size;
        desc.entryPointName = pipelineDesc.pixelShaderEntryPointName;
        desc.stage = nri::StageBits::FRAGMENT_SHADER;
    }

    nriPipelineDesc.shaders = shaderStages.data();
    nriPipelineDesc.shaderNum = (uint32_t)shaderStages.size();
    NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, nriPipelineDesc, m_NriPipelines.emplace_back()));
}

void OmmBakerGpuIntegration::CreateComputePipeline(uint32_t id, const ommGpuPipelineInfoDesc* pipelineInfo) {
    const ommGpuComputePipelineDesc& pipelineDesc = pipelineInfo->pipelines[id].compute;

    std::vector<nri::DescriptorRangeDesc> descriptorRangeDescs(pipelineDesc.descriptorRangeNum + 2);
    FillDescriptorRangeDescs(pipelineDesc.descriptorRangeNum, pipelineDesc.descriptorRanges, descriptorRangeDescs.data());

    nri::DescriptorRangeDesc& staticSamplersRange = descriptorRangeDescs[pipelineDesc.descriptorRangeNum + 0];
    staticSamplersRange.baseRegisterIndex = 0;
    staticSamplersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamplersRange.descriptorType = nri::DescriptorType::SAMPLER;
    staticSamplersRange.flags = nri::DescriptorRangeBits::NONE;
    staticSamplersRange.shaderStages = nri::StageBits::ALL;

    nri::DescriptorRangeDesc& constantBufferRange = descriptorRangeDescs[pipelineDesc.descriptorRangeNum + 1];
    constantBufferRange.baseRegisterIndex = pipelineInfo->globalConstantBufferDesc.registerIndex;
    constantBufferRange.descriptorNum = 1;
    constantBufferRange.descriptorType = nri::DescriptorType::CONSTANT_BUFFER;
    constantBufferRange.flags = nri::DescriptorRangeBits::NONE;
    constantBufferRange.shaderStages = nri::StageBits::ALL;

    nri::DescriptorSetDesc descriptorSetDescs = {};
    descriptorSetDescs.rangeNum = (uint32_t)descriptorRangeDescs.size();
    descriptorSetDescs.ranges = descriptorRangeDescs.data();

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = &descriptorSetDescs;
    layoutDesc.descriptorSetNum = 1;
    layoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::RootConstantDesc pushConstantDesc = {};
    {
        pushConstantDesc.registerIndex = pipelineInfo->localConstantBufferDesc.registerIndex;
        pushConstantDesc.size = pipelineInfo->localConstantBufferDesc.maxDataSize;
        pushConstantDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        layoutDesc.rootConstants = &pushConstantDesc;
        layoutDesc.rootConstantNum = 1;
    }
    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, layoutDesc, m_NriPipelineLayouts.emplace_back()));

    nri::ComputePipelineDesc nriPipelineDesc = {};
    nriPipelineDesc.pipelineLayout = m_NriPipelineLayouts.back();
    nriPipelineDesc.shader.bytecode = pipelineDesc.computeShader.data;
    nriPipelineDesc.shader.size = pipelineDesc.computeShader.size;
    nriPipelineDesc.shader.entryPointName = pipelineDesc.shaderEntryPointName;
    nriPipelineDesc.shader.stage = nri::StageBits::COMPUTE_SHADER;
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, nriPipelineDesc, m_NriPipelines.emplace_back()));
}

inline void FillSamplerDesc(nri::SamplerDesc& nriDesc, const ommGpuStaticSamplerDesc& ommDesc) {
    nriDesc = {};
    nriDesc.addressModes.u = GetNriAddressMode(ommDesc.desc.addressingMode);
    nriDesc.addressModes.v = GetNriAddressMode(ommDesc.desc.addressingMode);
    nriDesc.filters.mag = GetNriFilterMode(ommDesc.desc.filter);
    nriDesc.filters.min = GetNriFilterMode(ommDesc.desc.filter);
    nriDesc.filters.mip = GetNriFilterMode(ommDesc.desc.filter);
    nriDesc.mipMax = 16.0f;
    nriDesc.compareOp = nri::CompareOp::NONE;
}

void OmmBakerGpuIntegration::CreateSamplers(const ommGpuPipelineInfoDesc* pipelinesInfo) {
    for (uint32_t i = 0; i < pipelinesInfo->staticSamplersNum; ++i) {
        nri::SamplerDesc samplerDesc = {};
        const ommGpuStaticSamplerDesc& ommDesc = pipelinesInfo->staticSamplers[i];
        FillSamplerDesc(samplerDesc, ommDesc);
        nri::Descriptor* descriptor = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Samplers.push_back(descriptor);
    }
}

void OmmBakerGpuIntegration::CreateTextures(uint32_t pipelineNum) {
    m_ColorDescriptorPerPipeline.resize(pipelineNum, nullptr);

    { // create debug texture
        constexpr uint16_t maxTexSize = 8042;
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
        textureDesc.layerNum = 1;
        textureDesc.format = m_DebugTexFormat;
        textureDesc.width = maxTexSize;
        textureDesc.height = maxTexSize;
        textureDesc.depth = 1;
        textureDesc.sampleNum = 1;
        textureDesc.mipNum = 1;
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_DebugTexture));

        nri::ResourceGroupDesc resourceGrpoupDesc = {};
        resourceGrpoupDesc.textureNum = 1;
        resourceGrpoupDesc.textures = &m_DebugTexture;
        resourceGrpoupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGrpoupDesc, &m_DebugTextureMemory));

        nri::Texture2DViewDesc textureViewDesc = {};
        textureViewDesc.viewType = nri::Texture2DViewType::COLOR_ATTACHMENT;
        textureViewDesc.mipNum = 1;
        textureViewDesc.mipOffset = 0;
        textureViewDesc.format = m_DebugTexFormat;
        textureViewDesc.texture = m_DebugTexture;
        NRI.CreateTexture2DView(textureViewDesc, m_DebugTextureDescriptor);
    }
}

void OmmBakerGpuIntegration::CreatePipelines(const ommGpuPipelineInfoDesc* pipelinesInfo) {
    for (uint32_t i = 0; i < pipelinesInfo->pipelineNum; ++i) {
        const ommGpuPipelineDesc& ommPipelineDesc = pipelinesInfo->pipelines[i];
        switch (ommPipelineDesc.type) {
            case ommGpuPipelineType_Compute:
                CreateComputePipeline(i, pipelinesInfo);
                break;
            case ommGpuPipelineType_Graphics:
                CreateGraphicsPipeline(i, pipelinesInfo);
                break;
            default:
                printf("[FAIL] Invalid ommGpuPipelineType\n");
                std::abort();
        }
    }
}

void OmmBakerGpuIntegration::CreateStaticResources(nri::Queue* commandQueue) {
    ommGpuResourceType staticResources[] = {ommGpuResourceType_STATIC_INDEX_BUFFER, ommGpuResourceType_STATIC_VERTEX_BUFFER};
    nri::BufferUsageBits usageBits[] = {nri::BufferUsageBits::INDEX_BUFFER, nri::BufferUsageBits::VERTEX_BUFFER};
    nri::AccessBits nexAccessBits[] = {nri::AccessBits::INDEX_BUFFER, nri::AccessBits::VERTEX_BUFFER};
    nri::BufferUploadDesc bufferUploadDescs[(uint32_t)GpuStaticResources::Count];
    std::vector<uint8_t> uploadData[(uint32_t)GpuStaticResources::Count];

    for (uint32_t i = 0; i < (uint32_t)GpuStaticResources::Count; ++i) {
        size_t outSize = 0;
        ommGpuGetStaticResourceData(staticResources[i], nullptr, &outSize);
        uploadData[i].resize(outSize);
        ommGpuGetStaticResourceData(staticResources[i], uploadData[i].data(), &outSize);

        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = outSize;
        bufferDesc.usage = usageBits[i];
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_StaticBuffers[i].buffer));

        nri::BufferUploadDesc& uploadDesc = bufferUploadDescs[i];
        uploadDesc.buffer = m_StaticBuffers[i].buffer;
        uploadDesc.data = &uploadData[i][0];
        uploadDesc.after.access = nri::AccessBits::NONE;
        uploadDesc.after.stages = nri::StageBits::ALL;
    }

    nri::Buffer* buffers[] = {m_StaticBuffers[0].buffer, m_StaticBuffers[1].buffer};
    nri::ResourceGroupDesc resourceGrpoupDesc = {};
    resourceGrpoupDesc.bufferNum = (uint32_t)GpuStaticResources::Count;
    resourceGrpoupDesc.buffers = buffers;
    resourceGrpoupDesc.memoryLocation = nri::MemoryLocation::DEVICE;

    size_t currentMemoryAllocSize = m_NriStaticMemories.size();
    uint32_t allocRequestNum = NRI.CalculateAllocationNumber(*m_Device, resourceGrpoupDesc);
    m_NriStaticMemories.resize(currentMemoryAllocSize + allocRequestNum, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGrpoupDesc, m_NriStaticMemories.data() + currentMemoryAllocSize));
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*commandQueue, nullptr, 0, bufferUploadDescs, (uint32_t)GpuStaticResources::Count));
}

void FillDispatchConfigDesc(ommGpuDispatchConfigDesc& dispatchConfigDesc, const InputGeometryDesc& desc) {
    dispatchConfigDesc = ommGpuDispatchConfigDescDefault();

    const BakerInputs& inputs = desc.inputs;
    const BakerSettings& settings = desc.settings;

    dispatchConfigDesc.alphaTextureWidth = inputs.inTexture.width;
    dispatchConfigDesc.alphaTextureHeight = inputs.inTexture.height;
    dispatchConfigDesc.alphaTextureChannel = inputs.inTexture.alphaChannelId;

    dispatchConfigDesc.alphaMode = (ommAlphaMode)settings.alphaMode;
    dispatchConfigDesc.alphaCutoff = settings.alphaCutoff;

    dispatchConfigDesc.indexFormat = GetOmmIndexFormat(inputs.inIndexBuffer.format);
    dispatchConfigDesc.indexCount = (uint32_t)inputs.inIndexBuffer.numElements;
    dispatchConfigDesc.indexStrideInBytes = (uint32_t)inputs.inIndexBuffer.stride;

    dispatchConfigDesc.texCoordFormat = GetOmmTexcoordFormat(inputs.inUvBuffer.format);
    dispatchConfigDesc.texCoordStrideInBytes = (uint32_t)inputs.inUvBuffer.stride;
    dispatchConfigDesc.texCoordOffsetInBytes = (uint32_t)inputs.inUvBuffer.offsetInStruct;

    dispatchConfigDesc.runtimeSamplerDesc.addressingMode = GetOmmAddressingMode(settings.samplerAddressingMode);
    dispatchConfigDesc.runtimeSamplerDesc.filter = GetOmmFilterMode(settings.samplerFilterMode);
    dispatchConfigDesc.runtimeSamplerDesc.borderAlpha = settings.borderAlpha;

    dispatchConfigDesc.globalFormat = ommFormat(settings.globalOMMFormat);

    dispatchConfigDesc.maxSubdivisionLevel = (uint8_t)settings.maxSubdivisionLevel;
    dispatchConfigDesc.enableSubdivisionLevelBuffer = false; // TODO: make a var
    dispatchConfigDesc.maxScratchMemorySize = GetScratchMemoryBudget(settings.maxScratchMemorySize);
    dispatchConfigDesc.dynamicSubdivisionScale = settings.dynamicSubdivisionScale;
    dispatchConfigDesc.bakeFlags = GetBakeFlags(settings.bakeFlags);
    dispatchConfigDesc.maxOutOmmArraySize = uint32_t(~0);
}

inline uint32_t GetAlignedSize(uint32_t size, uint32_t alignment) {
    return (((size + alignment - 1) / alignment) * alignment);
}

void OmmBakerGpuIntegration::GetPrebuildInfo(InputGeometryDesc* geometryDesc, uint32_t geometryNum) {
    for (uint32_t i = 0; i < geometryNum; ++i) {
        InputGeometryDesc& desc = geometryDesc[i];
        ommGpuDispatchConfigDesc dispatchConfigDesc = ommGpuDispatchConfigDescDefault();

        FillDispatchConfigDesc(dispatchConfigDesc, desc);

        ommGpuPreDispatchInfo info = ommGpuPreDispatchInfoDefault();
        ommResult ommResult = ommGpuGetPreDispatchInfo(m_Pipeline, &dispatchConfigDesc, &info);
        if (ommResult != ommResult_SUCCESS) {
            printf("[FAIL] ommGpuGetPreBakeInfo()\n");
            std::abort();
        }

        PrebuildInfo& prebuildInfo = desc.outputs.prebuildInfo;
        prebuildInfo.arrayDataSize = info.outOmmArraySizeInBytes;
        prebuildInfo.descArraySize = info.outOmmDescSizeInBytes;
        prebuildInfo.indexBufferSize = info.outOmmIndexBufferSizeInBytes;
        prebuildInfo.ommDescArrayHistogramSize = info.outOmmArrayHistogramSizeInBytes;
        prebuildInfo.ommIndexHistogramSize = info.outOmmIndexHistogramSizeInBytes;
        prebuildInfo.postBuildInfoSize = info.outOmmPostDispatchInfoSizeInBytes;
        for (size_t j = 0; j < info.numTransientPoolBuffers; ++j)
            prebuildInfo.transientBufferSizes[j] = info.transientPoolBufferSizeInBytes[j];

        prebuildInfo.indexCount = info.outOmmIndexCount;
        prebuildInfo.indexFormat = GetNriIndexFormat(info.outOmmIndexBufferFormat);
    }
}

void OmmBakerGpuIntegration::AddGeometryToQueue(InputGeometryDesc* geometryDesc, uint32_t geometryNum) {
    m_GeometryQueue.resize(geometryNum);

    for (uint32_t i = 0; i < geometryNum; ++i) {
        GeometryQueueInstance& instance = m_GeometryQueue[i];
        instance.desc = &geometryDesc[i];

        FillDispatchConfigDesc(instance.dispatchConfigDesc, *instance.desc);

        ommGpuPreDispatchInfo info = ommGpuPreDispatchInfoDefault();
        ommResult ommResult = ommGpuGetPreDispatchInfo(m_Pipeline, &instance.dispatchConfigDesc, &info);
        if (ommResult != ommResult_SUCCESS) {
            printf("[FAIL][OMM][GPU] ommGpuGetPreDispatchInfo failed.\n");
            std::abort();
        }
    }
}

void OmmBakerGpuIntegration::UpdateGlobalConstantBuffer() {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    uint32_t newConstantBufferViewSize = GetAlignedSize(m_PipelineInfo->globalConstantBufferDesc.maxDataSize, uint32_t(deviceDesc.memoryAlignment.constantBufferOffset));
    uint32_t newConstantBufferSize = newConstantBufferViewSize * (uint32_t)m_GeometryQueue.size();

    if (m_ConstantBufferSize < newConstantBufferSize) {
        m_ConstantBufferSize = newConstantBufferSize;
        m_ConstantBufferViewStride = 0;
        if (m_ConstantBuffer)
            NRI.DestroyBuffer(m_ConstantBuffer);
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = m_ConstantBufferSize;
        bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_ConstantBuffer;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, &m_ConstantBufferHeap));
    }

    if (m_ConstantBufferViewStride < newConstantBufferViewSize) {
        m_ConstantBufferViewStride = newConstantBufferViewSize;
        if (m_ConstantBufferViews.empty() == false){
            for(auto& view : m_ConstantBufferViews){
                NRI.DestroyDescriptor(view);
            }
        }

        m_ConstantBufferViews.resize((uint32_t)m_GeometryQueue.size());
        nri::BufferViewDesc constantBufferViewDesc = {};
        constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
        constantBufferViewDesc.buffer = m_ConstantBuffer;
        constantBufferViewDesc.size = m_ConstantBufferViewStride;

        for(uint64_t i = 0; i < (uint64_t)m_GeometryQueue.size(); ++i) {
            constantBufferViewDesc.offset = m_ConstantBufferViewStride * i;
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(constantBufferViewDesc, m_ConstantBufferViews[i]));
        }
    }
}

inline uint64_t ComputeHash(const void* key, uint32_t len, uint32_t geometryId) {
    const uint8_t* p = (uint8_t*)key;
    uint64_t result = 14695981039346656037ull - geometryId;
    while (len--)
        result = (result ^ (*p++)) * 1099511628211ull;

    return result;
}

void OmmBakerGpuIntegration::UpdateDescriptorPool(uint32_t geometryId, const ommGpuDispatchChain* dispatchChain) {
    nri::DescriptorPool*& desctriptorPool = m_NriDescriptorPools[geometryId];
    if (desctriptorPool)
        NRI.DestroyDescriptorPool(desctriptorPool);

    nri::DescriptorPoolDesc desc = {};
    uint32_t dispatchNum = 0;
    uint32_t uniqueDescriptorSetNum = 0;
    for (uint32_t i = 0; i < dispatchChain->numDispatches; ++i) { // filter out labling events
        switch (dispatchChain->dispatches[i].type) {
            case ommGpuDispatchType_BeginLabel:
            case ommGpuDispatchType_EndLabel:
                break;
            default: {
                uint64_t hash = ComputeHash(dispatchChain->dispatches[i].compute.resources, dispatchChain->dispatches[i].compute.resourceNum * sizeof(ommGpuResource), geometryId);
                const auto& it = m_NriDescriptorSets.find(hash);
                if (it == m_NriDescriptorSets.end()) {
                    m_NriDescriptorSets.insert(std::make_pair(hash, nullptr));
                    ++uniqueDescriptorSetNum;

                    for (uint32_t j = 0; j < dispatchChain->dispatches[i].compute.resourceNum; ++j) {
                        const ommGpuResource& resource = dispatchChain->dispatches[i].compute.resources[j];
                        switch (resource.stateNeeded) {
                            case ommGpuDescriptorType_TextureRead:
                                ++desc.textureMaxNum;
                                break;
                            case ommGpuDescriptorType_BufferRead:
                                ++desc.bufferMaxNum;
                                break;
                            case ommGpuDescriptorType_RawBufferRead:
                                ++desc.structuredBufferMaxNum;
                                break;
                            case ommGpuDescriptorType_RawBufferWrite:
                                ++desc.storageStructuredBufferMaxNum;
                                break;
                            default:
                                break;
                        }
                    }
                }
                ++dispatchNum;
            }
        }
    }

    desc.descriptorSetMaxNum = uniqueDescriptorSetNum;
    desc.constantBufferMaxNum = dispatchNum;
    desc.samplerMaxNum = uniqueDescriptorSetNum * (uint32_t)m_Samplers.size();
    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, desc, desctriptorPool));
}

uint64_t CalculateDescriptorKey(uint32_t geometryId, const ommGpuResource& resource) {
    bool isTransientPool = resource.type == ommGpuResourceType_TRANSIENT_POOL_BUFFER;
    uint64_t key = isTransientPool ? 0 : geometryId + 1;
    key |= uint64_t(resource.type) << 32ull;
    key |= uint64_t(resource.stateNeeded) << 40ull;
    key |= uint64_t(resource.indexInPool) << 48ull;
    return key;
}

nri::Descriptor* OmmBakerGpuIntegration::GetDescriptor(const ommGpuResource& resource, uint32_t geometryId) {
    uint64_t key = CalculateDescriptorKey(geometryId, resource);
    nri::Descriptor* descriptor = nullptr;
    const auto& it = m_NriDescriptors.find(key);
    if (it == m_NriDescriptors.end()) {
        BakerInputs& inputs = m_GeometryQueue[geometryId].desc->inputs;
        bool isTexture = resource.stateNeeded == ommGpuDescriptorType_TextureRead;
        bool isRaw = (resource.stateNeeded == ommGpuDescriptorType_RawBufferRead) || (resource.stateNeeded == ommGpuDescriptorType_RawBufferWrite);
        if (isTexture) {
            nri::Texture2DViewDesc texDesc = {};
            texDesc.mipNum = 1;
            texDesc.mipOffset = nri::Dim_t(inputs.inTexture.mipOffset);
            texDesc.viewType = nri::Texture2DViewType::SHADER_RESOURCE_2D;
            texDesc.format = inputs.inTexture.format;
            texDesc.texture = inputs.inTexture.texture;
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texDesc, descriptor));
        } else {
            const BufferResource& buffer = GetBuffer(resource, geometryId);
            nri::BufferViewDesc bufferDesc = {};
            bufferDesc.buffer = buffer.buffer;
            bufferDesc.offset = buffer.offset;
            bufferDesc.format = isRaw ? nri::Format::UNKNOWN : buffer.format;
            bufferDesc.size = buffer.size - buffer.offset;
            bufferDesc.viewType = GetNriBufferViewType(resource.stateNeeded);
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferDesc, descriptor));
        }
        m_NriDescriptors.insert(std::make_pair(key, descriptor));
    } else
        descriptor = it->second;

    return descriptor;
}

void OmmBakerGpuIntegration::PerformResourceTransition(const ommGpuResource& resource, uint32_t geometryId, std::vector<nri::BufferBarrierDesc>& bufferBarriers) {
    if (resource.type == ommGpuResourceType_IN_ALPHA_TEXTURE)
        return;

    BufferResource& bufferResource = GetBuffer(resource, geometryId);
    nri::AccessBits currentState = bufferResource.state;
    nri::AccessBits requestedState = GetNriResourceState(resource.stateNeeded);

    if (currentState != requestedState) {
        nri::BufferBarrierDesc& barrier = bufferBarriers.emplace_back();
        barrier.before.access = currentState;
        barrier.before.stages = nri::StageBits::ALL;
        barrier.after.access = requestedState;
        barrier.after.stages = nri::StageBits::ALL;
        barrier.buffer = bufferResource.buffer;

        bufferResource.state = requestedState;
    }
}

nri::DescriptorSet* OmmBakerGpuIntegration::PrepareDispatch(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t pipelineIndex, uint32_t geometryId) {
    nri::PipelineLayout*& pipelineLayout = m_NriPipelineLayouts[pipelineIndex];

    // Descriptor set
    uint64_t hash = ComputeHash(resources, resourceNum * sizeof(ommGpuResource), geometryId);
    const auto& it = m_NriDescriptorSets.find(hash);
    nri::DescriptorSet* descriptorSet = nullptr;
    bool updateRanges = false;
    if (it->second == nullptr) {
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_NriDescriptorPools[geometryId], *pipelineLayout, 0, &descriptorSet, 1, 0));
        it->second = descriptorSet;
        updateRanges = true;
    } else
        descriptorSet = it->second;

    // process requested resources. prepare range updates. perform transitions
    std::vector<nri::Descriptor*> descriptors;
    descriptors.resize(resourceNum);

    std::vector<nri::UpdateDescriptorRangeDesc> rangeUpdateDescs;
    std::vector<nri::BufferBarrierDesc> bufferTransitions;
    nri::DescriptorType prevRangeType = nri::DescriptorType::MAX_NUM;
    for (uint32_t i = 0; i < resourceNum; ++i) {
        const ommGpuResource& resource = resources[i];
        nri::DescriptorType rangeType = GetNriDescriptorType(resource.stateNeeded);
        if (rangeType != prevRangeType) {
            nri::UpdateDescriptorRangeDesc nextRange = {};
            nextRange.descriptors = descriptors.data() + i;
            nextRange.rangeIndex = (uint32_t)rangeUpdateDescs.size();
            rangeUpdateDescs.push_back(nextRange);
            prevRangeType = rangeType;
        }

        nri::UpdateDescriptorRangeDesc& currentRange = rangeUpdateDescs.back();
        descriptors[i] = GetDescriptor(resources[i], geometryId);
        currentRange.descriptorNum += 1;
        currentRange.descriptorSet = descriptorSet;
        PerformResourceTransition(resource, geometryId, bufferTransitions);
    }

    nri::UpdateDescriptorRangeDesc& staticSamlersRange = rangeUpdateDescs.emplace_back();
    staticSamlersRange.descriptors = m_Samplers.data();
    staticSamlersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamlersRange.descriptorSet = descriptorSet;
    staticSamlersRange.baseDescriptor = 0;
    staticSamlersRange.rangeIndex = uint32_t(rangeUpdateDescs.size() - 1);

    nri::UpdateDescriptorRangeDesc& constantBufferRange = rangeUpdateDescs.emplace_back();
    constantBufferRange.descriptors = &m_ConstantBufferViews[geometryId];
    constantBufferRange.descriptorNum = 1;
    constantBufferRange.descriptorSet = descriptorSet;
    constantBufferRange.baseDescriptor = 0;
    constantBufferRange.rangeIndex = uint32_t(rangeUpdateDescs.size() - 1);

    if (updateRanges) {
        NRI.UpdateDescriptorRanges(rangeUpdateDescs.data(), (uint32_t) rangeUpdateDescs.size());
    }

    nri::BarrierDesc transitionBarriers = {};
    transitionBarriers.bufferNum = (uint32_t)bufferTransitions.size();
    transitionBarriers.buffers = bufferTransitions.data();
    if (transitionBarriers.bufferNum)
        NRI.CmdBarrier(commandBuffer, transitionBarriers);

    nri::BindPoint bindPoint = m_PipelineInfo->pipelines[pipelineIndex].type == ommGpuPipelineType::ommGpuPipelineType_Graphics ? nri::BindPoint::GRAPHICS : nri::BindPoint::COMPUTE; 
    NRI.CmdSetPipelineLayout(commandBuffer, bindPoint, *pipelineLayout);

    NRI.CmdSetPipeline(commandBuffer, *m_NriPipelines[pipelineIndex]);

    return descriptorSet;
}

void OmmBakerGpuIntegration::InsertUavBarriers(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t geometryId) {
    std::vector<nri::BufferBarrierDesc> uavBarriers;
    for (uint32_t i = 0; i < resourceNum; ++i) {
        if (resources[i].stateNeeded == ommGpuDescriptorType_RawBufferWrite) {
            nri::BufferBarrierDesc barrier = {
                GetBuffer(resources[i], geometryId).buffer,
                {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::ALL},
                {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::ALL},
            };
            uavBarriers.push_back(barrier);
        }
    }
    nri::BarrierDesc transition = {};
    transition.bufferNum = (uint32_t)uavBarriers.size();
    transition.buffers = uavBarriers.data();
    NRI.CmdBarrier(commandBuffer, transition);
}

void OmmBakerGpuIntegration::DispatchCompute(nri::CommandBuffer& commandBuffer, const ommGpuComputeDesc& desc, uint32_t geometryId) {
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);

    if (desc.localConstantBufferDataSize) {
        nri::SetRootConstantsDesc rootConstantsDesc = {};
        rootConstantsDesc.rootConstantIndex = 0;
        rootConstantsDesc.bindPoint = nri::BindPoint::INHERIT;
        rootConstantsDesc.data = desc.localConstantBufferData;
        rootConstantsDesc.size = desc.localConstantBufferDataSize;
        rootConstantsDesc.offset = 0;
        NRI.CmdSetRootConstants(commandBuffer, rootConstantsDesc);
    }

    {
        nri::SetDescriptorSetDesc setDescriptorSetDesc = {};
        setDescriptorSetDesc.bindPoint = nri::BindPoint::INHERIT;
        setDescriptorSetDesc.descriptorSet = descriptorSet;
        setDescriptorSetDesc.setIndex = 0;
        NRI.CmdSetDescriptorSet(commandBuffer, setDescriptorSetDesc);
    }

    nri::DispatchDesc dispatchDesc = {};
    dispatchDesc.x = desc.gridWidth;
    dispatchDesc.y = desc.gridHeight;
    dispatchDesc.z = 1;
    NRI.CmdDispatch(commandBuffer, dispatchDesc);
    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void OmmBakerGpuIntegration::DispatchComputeIndirect(nri::CommandBuffer& commandBuffer, const ommGpuComputeIndirectDesc& desc, uint32_t geometryId) {
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);

    if (desc.localConstantBufferDataSize){
        nri::SetRootConstantsDesc rootConstantsDesc = {};
        rootConstantsDesc.bindPoint = nri::BindPoint::INHERIT;
        rootConstantsDesc.data = desc.localConstantBufferData;
        rootConstantsDesc.size = desc.localConstantBufferDataSize;
        rootConstantsDesc.offset = 0;
        NRI.CmdSetRootConstants(commandBuffer, rootConstantsDesc);
    }

    {
        nri::SetDescriptorSetDesc setDescriptorSetDesc = {};
        setDescriptorSetDesc.bindPoint = nri::BindPoint::INHERIT;
        setDescriptorSetDesc.descriptorSet = descriptorSet;
        setDescriptorSetDesc.setIndex = 0;
        NRI.CmdSetDescriptorSet(commandBuffer, setDescriptorSetDesc);
    }

    BufferResource& argBuffer = GetBuffer(desc.indirectArg, geometryId);
    if (argBuffer.state != nri::AccessBits::ARGUMENT_BUFFER) {
        nri::BufferBarrierDesc bufferBarrier = {};
        bufferBarrier.buffer = argBuffer.buffer;
        bufferBarrier.before.access = argBuffer.state;
        bufferBarrier.before.stages = nri::StageBits::ALL;
        bufferBarrier.after.access = nri::AccessBits::ARGUMENT_BUFFER;
        bufferBarrier.after.stages = nri::StageBits::ALL;

        nri::BarrierDesc transition = {};
        transition.bufferNum = 1;
        transition.buffers = &bufferBarrier;
        NRI.CmdBarrier(commandBuffer, transition);

        argBuffer.state = nri::AccessBits::ARGUMENT_BUFFER;
    }
    NRI.CmdDispatchIndirect(commandBuffer, *argBuffer.buffer, desc.indirectArgByteOffset);
    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void OmmBakerGpuIntegration::DispatchDrawIndexedIndirect(nri::CommandBuffer& commandBuffer, const ommGpuDrawIndexedIndirectDesc& desc, uint32_t geometryId) {
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);

    if (desc.localConstantBufferDataSize) {
        nri::SetRootConstantsDesc rootConstantsDesc = {};
        rootConstantsDesc.bindPoint = nri::BindPoint::INHERIT;
        rootConstantsDesc.data = desc.localConstantBufferData;
        rootConstantsDesc.size = desc.localConstantBufferDataSize;
        rootConstantsDesc.offset = 0;
        NRI.CmdSetRootConstants(commandBuffer, rootConstantsDesc);
    }

    {
        nri::SetDescriptorSetDesc setDescriptorSetDesc = {};
        setDescriptorSetDesc.bindPoint = nri::BindPoint::INHERIT;
        setDescriptorSetDesc.descriptorSet = descriptorSet;
        setDescriptorSetDesc.setIndex = 0;
        NRI.CmdSetDescriptorSet(commandBuffer, setDescriptorSetDesc);
    }

    BufferResource& argBuffer = GetBuffer(desc.indirectArg, geometryId);
    if (argBuffer.state != nri::AccessBits::ARGUMENT_BUFFER) {
        nri::BufferBarrierDesc bufferBarrier = {};
        bufferBarrier.buffer = argBuffer.buffer;
        bufferBarrier.before.access = argBuffer.state;
        bufferBarrier.before.stages = nri::StageBits::ALL;
        bufferBarrier.after.access = nri::AccessBits::ARGUMENT_BUFFER;
        bufferBarrier.after.stages = nri::StageBits::ALL;

        nri::BarrierDesc transition = {};
        transition.bufferNum = 1;
        transition.buffers = &bufferBarrier;
        NRI.CmdBarrier(commandBuffer, transition);

        argBuffer.state = nri::AccessBits::ARGUMENT_BUFFER;
    }

    nri::AttachmentsDesc frameBuffer = {};
    frameBuffer.colors = &m_ColorDescriptorPerPipeline[desc.pipelineIndex];
    frameBuffer.colorNum = m_ColorDescriptorPerPipeline[desc.pipelineIndex] ? 1 : 0;
    const bool hasDebugTextureOutput = frameBuffer.colorNum > 0;

    if (hasDebugTextureOutput && m_DebugTextureState != nri::AccessBits::COLOR_ATTACHMENT) { // perform debug frame buffer transition
        nri::TextureBarrierDesc textureBarrierDesc = {};
        textureBarrierDesc.texture = m_DebugTexture;
        textureBarrierDesc.mipNum = 1;
        textureBarrierDesc.mipOffset = 0;
        textureBarrierDesc.layerOffset = 0;
        textureBarrierDesc.layerNum = 1;
        textureBarrierDesc.before.access = m_DebugTextureState;
        textureBarrierDesc.before.stages = nri::StageBits::ALL;
        textureBarrierDesc.after.access = nri::AccessBits::COLOR_ATTACHMENT;
        textureBarrierDesc.after.stages = nri::StageBits::ALL;

        nri::BarrierDesc barrier = {};
        barrier.textureNum = 1;
        barrier.textures = &textureBarrierDesc;

        NRI.CmdBarrier(commandBuffer, barrier);
        m_DebugTextureState = nri::AccessBits::COLOR_ATTACHMENT;
    }

    NRI.CmdBeginRendering(commandBuffer, frameBuffer);
    {
        BufferResource& indexBuffer = GetBuffer(desc.indexBuffer, geometryId);
        NRI.CmdSetIndexBuffer(commandBuffer, *indexBuffer.buffer, desc.indexBufferOffset, nri::IndexType::UINT32);

        uint64_t offset = desc.vertexBufferOffset;
        nri::VertexBufferDesc vertexBuffer = {};
        vertexBuffer.buffer = GetBuffer(desc.vertexBuffer, geometryId).buffer;
        vertexBuffer.offset = offset;
        vertexBuffer.stride = sizeof(uint32_t);
        NRI.CmdSetVertexBuffers(commandBuffer, 0, &vertexBuffer, 1);

        nri::Viewport viewport = {};
        viewport.x = desc.viewport.minWidth;
        viewport.y = desc.viewport.minHeight;
        viewport.width = desc.viewport.minHeight;
        viewport.height = desc.viewport.maxWidth;
        viewport.depthMin = 0.0f;
        viewport.depthMax = 1.0f;
        NRI.CmdSetViewports(commandBuffer, &viewport, 1);

        nri::Rect scissorRect = {};
        scissorRect.x = int16_t(desc.viewport.minWidth);
        scissorRect.y = int16_t(desc.viewport.minHeight);
        scissorRect.width = nri::Dim_t(desc.viewport.maxWidth);
        scissorRect.height = nri::Dim_t(desc.viewport.maxHeight);
        NRI.CmdSetScissors(commandBuffer, &scissorRect, 1);

        NRI.CmdDrawIndexedIndirect(commandBuffer, *argBuffer.buffer, desc.indirectArgByteOffset, 1, 20, nullptr, 0); // TODO: replace last constant with a GAPI related var
    }
    NRI.CmdEndRendering(commandBuffer);

    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void PostBakeBufferTransition(std::vector<nri::BufferBarrierDesc>& transition, BufferResource& buffer) {
    if (buffer.buffer && buffer.state != nri::AccessBits::COPY_SOURCE) {
        nri::BufferBarrierDesc barrier = {};
        barrier.buffer = buffer.buffer;
        barrier.before.access = buffer.state;
        barrier.before.stages = nri::StageBits::ALL;
        barrier.after.access = nri::AccessBits::COPY_SOURCE;
        barrier.after.stages = nri::StageBits::ALL;
        transition.push_back(barrier);
    }
}

void OmmBakerGpuIntegration::GenerateVisibilityMaskGPU(nri::CommandBuffer& commandBuffer, uint32_t geometryId) {
    GeometryQueueInstance& instance = m_GeometryQueue[geometryId];
    ommGpuDispatchConfigDesc& dispatchConfigDesc = instance.dispatchConfigDesc;

    const ommGpuDispatchChain* dispatchChain = nullptr;
    ommGpuDispatch(m_Pipeline, &dispatchConfigDesc, &dispatchChain);

    // Update and set descriptor pool
    UpdateDescriptorPool(geometryId, dispatchChain);
    NRI.CmdSetDescriptorPool(commandBuffer, *m_NriDescriptorPools[geometryId]);

    // Upload constants
    if (dispatchChain->globalCBufferDataSize) {
        void* data = NRI.MapBuffer(*m_ConstantBuffer, m_ConstantBufferViewStride * (uint64_t)geometryId, dispatchChain->globalCBufferDataSize);
        memcpy(data, dispatchChain->globalCBufferData, dispatchChain->globalCBufferDataSize);
        NRI.UnmapBuffer(*m_ConstantBuffer);
    }

    for (uint32_t i = 0; i < dispatchChain->numDispatches; ++i) {
        const ommGpuDispatchDesc& dispacthDesc = dispatchChain->dispatches[i];
        switch (dispacthDesc.type) {
            case ommGpuDispatchType_BeginLabel:
                NRI.CmdBeginAnnotation(commandBuffer, dispacthDesc.beginLabel.debugName, 0);
                break;
            case ommGpuDispatchType_Compute: {
                const ommGpuComputeDesc& desc = dispacthDesc.compute;
                DispatchCompute(commandBuffer, desc, geometryId);
                break;
            }
            case ommGpuDispatchType_ComputeIndirect: {
                const ommGpuComputeIndirectDesc& desc = dispacthDesc.computeIndirect;
                DispatchComputeIndirect(commandBuffer, desc, geometryId);
                break;
            }
            case ommGpuDispatchType_DrawIndexedIndirect: {
                const ommGpuDrawIndexedIndirectDesc& desc = dispacthDesc.drawIndexedIndirect;
                DispatchDrawIndexedIndirect(commandBuffer, desc, geometryId);
                break;
            }
            case ommGpuDispatchType_EndLabel:
                NRI.CmdEndAnnotation(commandBuffer);
                break;
            default:
                break;
        }
    }

    BakerOutputs& outputs = instance.desc->outputs;
    BakerInputs& inputs = instance.desc->inputs;
    std::vector<nri::BufferBarrierDesc> outputBuffersTransition = {};
    PostBakeBufferTransition(outputBuffersTransition, outputs.outArrayData);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outDescArray);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outIndexBuffer);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outArrayHistogram);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outIndexHistogram);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outPostBuildInfo);

    for (size_t i = 0; i < OMM_MAX_TRANSIENT_POOL_BUFFERS; ++i)
        PostBakeBufferTransition(outputBuffersTransition, inputs.inTransientPool[i]);

    nri::BarrierDesc transitionBarriers = {};
    transitionBarriers.bufferNum = (uint32_t)outputBuffersTransition.size();
    transitionBarriers.buffers = outputBuffersTransition.data();

    if (transitionBarriers.bufferNum)
        NRI.CmdBarrier(commandBuffer, transitionBarriers);
}

void OmmBakerGpuIntegration::Bake(nri::CommandBuffer& commandBuffer, InputGeometryDesc* geometryDesc, uint32_t geometryNum) {
    if (!geometryNum)
        return;

    AddGeometryToQueue(geometryDesc, geometryNum);
    UpdateGlobalConstantBuffer();
    m_NriDescriptorPools.resize(geometryNum);

    for (uint32_t i = 0; i < geometryNum; ++i)
        GenerateVisibilityMaskGPU(commandBuffer, i);

    m_GeometryQueue.clear();
}

void OmmBakerGpuIntegration::ReleaseTemporalResources() {
    m_GeometryQueue.resize(0);
    m_GeometryQueue.shrink_to_fit();
    m_NriDescriptorSets.clear();

    for (auto it = m_NriDescriptors.begin(); it != m_NriDescriptors.end();) {
        if (it->second)
            NRI.DestroyDescriptor(it->second);
        it = m_NriDescriptors.erase(it);
    }

    for (auto& pool : m_NriDescriptorPools) {
        if (pool) {
            NRI.DestroyDescriptorPool(pool);
            pool = nullptr;
        }
    }
    m_NriDescriptorPools.resize(0);
    m_NriDescriptorPools.shrink_to_fit();

    if (m_ConstantBuffer)
        NRI.DestroyBuffer(m_ConstantBuffer);
    for(auto& view : m_ConstantBufferViews)
        NRI.DestroyDescriptor(view);
    if (m_ConstantBufferHeap)
        NRI.FreeMemory(m_ConstantBufferHeap);
    m_ConstantBufferViewStride = m_ConstantBufferSize = 0;
    m_ConstantBuffer = nullptr;
    m_ConstantBufferHeap = nullptr;
    m_ConstantBufferViews.resize(0);
    m_ConstantBufferViews.shrink_to_fit();
}

void OmmBakerGpuIntegration::Destroy() {
    if (m_DebugTextureDescriptor) {
        NRI.DestroyDescriptor(m_DebugTextureDescriptor);
        m_DebugTextureDescriptor = nullptr;
    }
    if (m_DebugTexture) {
        NRI.DestroyTexture(m_DebugTexture);
        m_DebugTexture = nullptr;
    }
    if (m_DebugTextureMemory) {
        NRI.FreeMemory(m_DebugTextureMemory);
        m_DebugTextureMemory = nullptr;
    }
    m_ColorDescriptorPerPipeline.resize(0);
    m_ColorDescriptorPerPipeline.shrink_to_fit();

    for (auto& sampler : m_Samplers)
        if (sampler)
            NRI.DestroyDescriptor(sampler);

    for (auto& pipeline : m_NriPipelines)
        if (pipeline)
            NRI.DestroyPipeline(pipeline);

    for (auto& layout : m_NriPipelineLayouts)
        if (layout)
            NRI.DestroyPipelineLayout(layout);

    for (uint32_t i = 0; i < (uint32_t)GpuStaticResources::Count; ++i) {
        if (m_StaticBuffers[i].buffer)
            NRI.DestroyBuffer(m_StaticBuffers[i].buffer);
    }

    for (auto& memory : m_NriStaticMemories)
        if (memory)
            NRI.FreeMemory(memory);

    ommGpuDestroyPipeline(m_GpuBaker, m_Pipeline);
    ommDestroyBaker(m_GpuBaker);
}