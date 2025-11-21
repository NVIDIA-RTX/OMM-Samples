// © 2022 NVIDIA Corporation
#include <future>
#include <map>
#include <set>
#include "VisibilityMasks/OmmHelper.h"

#include "NRIFramework.h"

#include "Extensions/NRIRayTracing.h"
#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIWrapperVK.h"

// NRD and NRI-based integration
#include "NRD.h"
#include "NRDIntegration.hpp"

#include "../Detex/detex.h"
#include "Profiler/NriProfiler.hpp"

#ifdef _WIN32
#    undef APIENTRY
#    include <windows.h> // SetForegroundWindow, GetConsoleWindow
#endif

//=================================================================================
// Settings
//=================================================================================

// NRD mode and other shared settings are here
#include "../Shaders/Include/Shared.hlsli"

constexpr uint32_t MAX_ANIMATED_INSTANCE_NUM = 512;
constexpr auto BLAS_RIGID_MESH_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE | nri::AccelerationStructureBits::ALLOW_COMPACTION;
constexpr auto TLAS_BUILD_BITS = nri::AccelerationStructureBits::PREFER_FAST_TRACE;
constexpr float ACCUMULATION_TIME = 0.33f;     // seconds
constexpr float NEAR_Z = 0.001f;               // m
constexpr float GLASS_THICKNESS = 0.002f;      // m
constexpr float CAMERA_BACKWARD_OFFSET = 0.0f; // m, 3rd person camera offset
constexpr float NIS_SHARPNESS = 0.2f;
constexpr bool CAMERA_RELATIVE = true;
constexpr bool ALLOW_BLAS_MERGING = true;
constexpr bool ALLOW_HDR = NRIF_PLATFORM == NRIF_WINDOWS; // use "WIN + ALT + B" to switch HDR mode
constexpr bool USE_LOW_PRECISION_FP_FORMATS = true;       // saves a bit of memory and performance
constexpr bool USE_DLSS_TNN = false;                      // replace CNN (legacy) with TNN (better)
constexpr nri::UpscalerType upscalerType = nri::UpscalerType::DLSR;
constexpr int32_t MAX_HISTORY_FRAME_NUM = (int32_t)std::min(60u, std::min(nrd::REBLUR_MAX_HISTORY_FRAME_NUM, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
constexpr uint32_t TEXTURES_PER_MATERIAL = 4;
constexpr uint32_t MAX_TEXTURE_TRANSITIONS_NUM = 32;
constexpr uint32_t DYNAMIC_CONSTANT_BUFFER_SIZE = 1024 * 1024; // 1MB

#if (SIGMA_TRANSLUCENCY == 1)
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW_TRANSLUCENCY
#else
#    define SIGMA_VARIANT nrd::Denoiser::SIGMA_SHADOW
#endif

//=================================================================================
// Important tests, sensitive to regressions or just testing base functionality
//=================================================================================

const std::vector<uint32_t> interior_checkMeTests = {{1, 3, 6, 8, 9, 10, 12, 13, 14, 23, 27, 28, 29, 31, 32, 35, 43, 44, 47, 53,
    59, 60, 62, 67, 75, 76, 79, 81, 95, 96, 107, 109, 111, 110, 114, 120, 124,
    126, 127, 132, 133, 134, 139, 140, 142, 145, 148, 150, 155, 156, 157, 160,
    161, 162, 164, 168, 169, 171, 172, 173, 174}};

//=================================================================================
// Tests, where IQ improvement would be "nice to have"
//=================================================================================

const std::vector<uint32_t> REBLUR_interior_improveMeTests = {{108, 110, 153, 174, 191, 192}};

const std::vector<uint32_t> RELAX_interior_improveMeTests = {{114, 144, 148, 156, 159}};

const std::vector<uint32_t> DLRR_interior_improveMeTests = {{
    1, 6, 159,   // snappy specular tracking
    4, 181,      // boily reaction to importance sampling
    62, 98, 112, // diffuse missing details and ghosting
    185, 186,    // missing material details (low confidence reprojection)
    220,         // patterns
    221,         // ortho
    222,         // diffuse darkening
}};

// TODO: add tests for SIGMA, active when "Shadow" visualization is on

//=================================================================================

// UI
#define UI_YELLOW            ImVec4(1.0f, 0.9f, 0.0f, 1.0f)
#define UI_GREEN             ImVec4(0.5f, 0.9f, 0.0f, 1.0f)
#define UI_RED               ImVec4(1.0f, 0.1f, 0.0f, 1.0f)
#define UI_HEADER            ImVec4(0.7f, 1.0f, 0.7f, 1.0f)
#define UI_HEADER_BACKGROUND ImVec4(0.7f * 0.3f, 1.0f * 0.3f, 0.7f * 0.3f, 1.0f)
#define UI_DEFAULT           ImGui::GetStyleColorVec4(ImGuiCol_Text)

enum class AccelerationStructure : uint32_t {
    TLAS_World,
    TLAS_Emissive,

    BLAS_MergedOpaque,
    BLAS_MergedTransparent,
    BLAS_MergedEmissive,
    BLAS_Other
};

enum class Buffer : uint32_t {
    InstanceData,
    PrimitiveData,
    SharcHashEntries,
    SharcAccumulated,
    SharcResolved,
    WorldScratch,
    LightScratch,
};

enum class Texture : uint32_t {
    ViewZ,
    Mv,
    Normal_Roughness,
    PsrThroughput,
    BaseColor_Metalness,
    DirectLighting,
    DirectEmission,
    Shadow,
    Diff,
    Spec,
    Unfiltered_Penumbra,
    Unfiltered_Diff,
    Unfiltered_Spec,
    Unfiltered_Translucency,
    Validation,
    Composed,

    // History
    ComposedDiff,
    ComposedSpec_ViewZ,
    TaaHistory,
    TaaHistoryPrev,

    // RR guides
    RRGuide_DiffAlbedo,
    RRGuide_SpecAlbedo,
    RRGuide_SpecHitDistance,
    RRGuide_Normal_Roughness, // only RGBA16f encoding is supported

    // Output resolution
    DlssOutput,
    PreFinal,

    // Window resolution
    Final,

    // SH
#if (NRD_MODE == SH)
    Unfiltered_DiffSh,
    Unfiltered_SpecSh,
    DiffSh,
    SpecSh,
#endif

    // Read-only
    MaterialTextures,
};

enum class Pipeline : uint32_t {
    SharcUpdate,
    SharcResolve,
    TraceOpaque,
    Composition,
    TraceTransparent,
    Taa,
    Final,
    DlssBefore,
    DlssAfter,
};

enum class Descriptor : uint32_t {
    World_AccelerationStructure,
    Light_AccelerationStructure,

    Constant_Buffer,
    InstanceData_Buffer,
    PrimitiveData_Buffer,
    PrimitiveData_StorageBuffer,
    SharcHashEntries_StorageBuffer,
    SharcAccumulated_StorageBuffer,
    SharcResolved_StorageBuffer,

    ViewZ_Texture,
    ViewZ_StorageTexture,
    Mv_Texture,
    Mv_StorageTexture,
    Normal_Roughness_Texture,
    Normal_Roughness_StorageTexture,
    PsrThroughput_Texture,
    PsrThroughput_StorageTexture,
    BaseColor_Metalness_Texture,
    BaseColor_Metalness_StorageTexture,
    DirectLighting_Texture,
    DirectLighting_StorageTexture,
    DirectEmission_Texture,
    DirectEmission_StorageTexture,
    Shadow_Texture,
    Shadow_StorageTexture,
    Diff_Texture,
    Diff_StorageTexture,
    Spec_Texture,
    Spec_StorageTexture,
    Unfiltered_Penumbra_Texture,
    Unfiltered_Penumbra_StorageTexture,
    Unfiltered_Diff_Texture,
    Unfiltered_Diff_StorageTexture,
    Unfiltered_Spec_Texture,
    Unfiltered_Spec_StorageTexture,
    Unfiltered_Translucency_Texture,
    Unfiltered_Translucency_StorageTexture,
    Validation_Texture,
    Validation_StorageTexture,
    Composed_Texture,
    Composed_StorageTexture,

    // History
    ComposedDiff_Texture,
    ComposedDiff_StorageTexture,
    ComposedSpec_ViewZ_Texture,
    ComposedSpec_ViewZ_StorageTexture,
    TaaHistory_Texture,
    TaaHistory_StorageTexture,
    TaaHistoryPrev_Texture,
    TaaHistoryPrev_StorageTexture,

    // RR guides
    RRGuide_DiffAlbedo_Texture,
    RRGuide_DiffAlbedo_StorageTexture,
    RRGuide_SpecAlbedo_Texture,
    RRGuide_SpecAlbedo_StorageTexture,
    RRGuide_SpecHitDistance_Texture,
    RRGuide_SpecHitDistance_StorageTexture,
    RRGuide_Normal_Roughness_Texture,
    RRGuide_Normal_Roughness_StorageTexture,

    // Output resolution
    DlssOutput_Texture,
    DlssOutput_StorageTexture,
    PreFinal_Texture,
    PreFinal_StorageTexture,

    // Window resolution
    Final_Texture,
    Final_StorageTexture,

    // SH
#if (NRD_MODE == SH)
    Unfiltered_DiffSh_Texture,
    Unfiltered_DiffSh_StorageTexture,
    Unfiltered_SpecSh_Texture,
    Unfiltered_SpecSh_StorageTexture,
    DiffSh_Texture,
    DiffSh_StorageTexture,
    SpecSh_Texture,
    SpecSh_StorageTexture,
#endif

    // Read-only
    MaterialTextures,
};

enum class DescriptorSet : uint32_t {
    // SET_OTHER
    TraceOpaque,
    Composition,
    TraceTransparent,
    TaaPing,
    TaaPong,
    Final,
    DlssBefore,
    DlssAfter,

    // SET_RAY_TRACING
    RayTracing, // must be first after "SET_OTHER"

    // SET_SHARC
    Sharc,
};

// NRD sample doesn't use several instances of the same denoiser in one NRD instance (like REBLUR_DIFFUSE x 3),
// thus we can use fields of "nrd::Denoiser" enum as unique identifiers
#define NRD_ID(x) nrd::Identifier(nrd::Denoiser::x)

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

struct Settings {
    double motionStartTime = 0.0;

    float maxFps = 60.0f;
    float camFov = 90.0f;
    float sunAzimuth = -147.0f;
    float sunElevation = 45.0f;
    float sunAngularDiameter = 0.533f;
    float exposure = 80.0f;
    float roughnessOverride = 0.0f;
    float metalnessOverride = 0.0f;
    float emissionIntensity = 1.0f;
    float debug = 0.0f;
    float meterToUnitsMultiplier = 1.0f;
    float emulateMotionSpeed = 1.0f;
    float animatedObjectScale = 1.0f;
    float separator = 0.0f;
    float animationProgress = 0.0f;
    float animationSpeed = 0.0f;
    float hitDistScale = 3.0f;
    float unused1 = 0.0f;
    float resolutionScale = 1.0f;
    float sharpness = 0.15f;

    int32_t maxAccumulatedFrameNum = 31;
    int32_t maxFastAccumulatedFrameNum = 7;
    int32_t onScreen = 0;
    int32_t forcedMaterial = 0;
    int32_t animatedObjectNum = 5;
    uint32_t activeAnimation = 0;
    int32_t motionMode = 0;
    int32_t denoiser = DENOISER_REBLUR;
    int32_t rpp = 1;
    int32_t bounceNum = 1;
    int32_t tracingMode = 0;
    int32_t mvType = 0;

    bool cameraJitter = true;
    bool limitFps = false;
    bool SHARC = true;
    bool PSR = false;
    bool indirectDiffuse = true;
    bool indirectSpecular = true;
    bool normalMap = true;
    bool TAA = true;
    bool animatedObjects = false;
    bool animateScene = false;
    bool animateSun = false;
    bool nineBrothers = false;
    bool blink = false;
    bool pauseAnimation = true;
    bool emission = true;
    bool linearMotion = true;
    bool emissiveObjects = false;
    bool importanceSampling = true;
    bool specularLobeTrimming = true;
    bool ortho = false;
    bool adaptiveAccumulation = true;
    bool usePrevFrame = true;
    bool windowAlignment = true;
    bool boost = false;
    bool SR = false;
    bool RR = false;
#pragma region[ OmmSample specific ]
    bool highLightAhs = true;
    bool ahsDynamicMipSelection = true;
#pragma endregion
};

struct DescriptorDesc {
    const char* debugName;
    void* resource;
    nri::Format format;
    nri::TextureUsageBits textureUsage;
    nri::BufferUsageBits bufferUsage;
    bool isArray;
};

struct TextureState {
    Texture texture;
    nri::AccessLayoutStage after;
};

struct AnimatedInstance {
    float3 basePosition;
    float3 rotationAxis;
    float3 elipseAxis;
    float durationSec = 5.0f;
    float progressedSec = 0.0f;
    uint32_t instanceID = 0;
    bool reverseRotation = true;
    bool reverseDirection = true;

    float4x4 Animate(float elapsedSeconds, float scale, float3& position) {
        float angle = progressedSec / durationSec;
        angle = Pi(angle * 2.0f - 1.0f);

        float3 localPosition;
        localPosition.x = cos(reverseDirection ? -angle : angle);
        localPosition.y = sin(reverseDirection ? -angle : angle);
        localPosition.z = localPosition.y;

        position = basePosition + localPosition * elipseAxis * scale;

        float4x4 transform;
        transform.SetupByRotation(reverseRotation ? -angle : angle, rotationAxis);
        transform.AddScale(scale);

        progressedSec = fmod(progressedSec + elapsedSeconds, durationSec);

        return transform;
    }
};

static inline nri::TextureBarrierDesc TextureBarrierFromUnknown(nri::Texture* texture, nri::AccessLayoutStage after) {
    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
    textureBarrier.before.access = nri::AccessBits::NONE;
    textureBarrier.before.layout = nri::Layout::UNDEFINED;
    textureBarrier.before.stages = nri::StageBits::NONE;
    textureBarrier.after = after;

    return textureBarrier;
}

static inline nri::TextureBarrierDesc TextureBarrierFromState(nri::TextureBarrierDesc& prevState, nri::AccessLayoutStage after) {
    prevState.before = prevState.after;
    prevState.after = after;

    return prevState;
}

#pragma region[ OmmSample specific ]
struct AlphaTestedGeometry {
    ommhelper::OmmBakeGeometryDesc bakeDesc;
    ommhelper::MaskedGeometryBuildDesc buildDesc;

    nri::Buffer* positions;
    nri::Buffer* uvs;
    nri::Buffer* indices;

    nri::Texture* alphaTexture;   // on gpu
    utils::Texture* utilsTexture; // on cpu

    std::vector<uint8_t> indexData;
    std::vector<uint8_t> uvData;

    uint64_t positionBufferSize;
    uint64_t positionOffset;
    uint64_t uvBufferSize;
    uint64_t uvOffset;
    uint64_t indexBufferSize;
    uint64_t indexOffset;

    uint32_t meshIndex;
    uint32_t materialIndex;

    const nri::Format vertexFormat = nri::Format::RGB32_SFLOAT;
    const nri::Format uvFormat = nri::Format::RG32_SFLOAT;
    const nri::Format indexFormat = nri::Format::R16_UINT;
};

struct OmmGpuBakerPrebuildMemoryStats {
    size_t total;
    size_t outputMaxSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
    size_t outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
    size_t maxTransientBufferSizes[OMM_MAX_TRANSIENT_POOL_BUFFERS];
};

struct OmmBatch {
    size_t offset;
    size_t count;
};
#pragma endregion

class Sample : public SampleBase {
public:
    inline Sample() {
#pragma region[ OmmSample specific ]
        m_SceneFile = "Bistro/BistroExterior.gltf";
        m_OutputResolution = {1920, 1080};
#pragma endregion
    }

    ~Sample();

    inline float GetDenoisingRange() const {
        return 4.0f * m_Scene.aabb.GetRadius();
    }

    inline bool IsDlssEnabled() const {
        return m_Settings.SR || m_Settings.RR;
    }

    inline nri::Texture*& Get(Texture index) {
        return m_Textures[(uint32_t)index];
    }

    inline nri::TextureBarrierDesc& GetState(Texture index) {
        return m_TextureStates[(uint32_t)index];
    }

    inline nri::Buffer*& Get(Buffer index) {
        return m_Buffers[(uint32_t)index];
    }

    inline nri::Pipeline*& Get(Pipeline index) {
        return m_Pipelines[(uint32_t)index];
    }

    inline nri::Descriptor*& Get(Descriptor index) {
        return m_Descriptors[(uint32_t)index];
    }

    inline nri::DescriptorSet*& Get(DescriptorSet index) {
        return m_DescriptorSets[(uint32_t)index];
    }

    inline nri::AccelerationStructure*& Get(AccelerationStructure index) {
        return m_AccelerationStructures[(uint32_t)index];
    }

    inline nrd::Resource GetNrdResource(Texture index) {
        nri::TextureBarrierDesc* textureState = &m_TextureStates[(uint32_t)index];

        nrd::Resource resource = {};
        resource.state = textureState->after;
        resource.userArg = textureState;
            resource.nri.texture = textureState->texture;

        return resource;
    }

    inline void Denoise(const nrd::Identifier* denoisers, uint32_t denoiserNum, nri::CommandBuffer& commandBuffer) {
        // Fill resource snapshot
        nrd::ResourceSnapshot resourceSnapshot = {};
        {
            resourceSnapshot.restoreInitialState = false;

            // Common
            resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, GetNrdResource(Texture::Mv));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, GetNrdResource(Texture::Normal_Roughness));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, GetNrdResource(Texture::ViewZ));

            // (Optional) Validation
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_VALIDATION, GetNrdResource(Texture::Validation));

            // Diffuse
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, GetNrdResource(Texture::Unfiltered_Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, GetNrdResource(Texture::Diff));

            // Specular
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, GetNrdResource(Texture::Unfiltered_Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, GetNrdResource(Texture::Spec));

#if (NRD_MODE == SH)
            // Diffuse SH
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_SH0, GetNrdResource(Texture::Unfiltered_Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_SH1, GetNrdResource(Texture::Unfiltered_DiffSh));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_SH0, GetNrdResource(Texture::Diff));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_SH1, GetNrdResource(Texture::DiffSh));

            // Specular SH
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_SH0, GetNrdResource(Texture::Unfiltered_Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_SH1, GetNrdResource(Texture::Unfiltered_SpecSh));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_SH0, GetNrdResource(Texture::Spec));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_SH1, GetNrdResource(Texture::SpecSh));
#endif

            // SIGMA
            resourceSnapshot.SetResource(nrd::ResourceType::IN_PENUMBRA, GetNrdResource(Texture::Unfiltered_Penumbra));
            resourceSnapshot.SetResource(nrd::ResourceType::IN_TRANSLUCENCY, GetNrdResource(Texture::Unfiltered_Translucency));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, GetNrdResource(Texture::Shadow));

            // REFERENCE
            resourceSnapshot.SetResource(nrd::ResourceType::IN_SIGNAL, GetNrdResource(Texture::Composed));
            resourceSnapshot.SetResource(nrd::ResourceType::OUT_SIGNAL, GetNrdResource(Texture::Composed));
        }

        // Denoise
            m_NRD.Denoise(denoisers, denoiserNum, commandBuffer, resourceSnapshot);

        // Retrieve state
        if (!resourceSnapshot.restoreInitialState) {
            for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
                nri::TextureBarrierDesc* state = (nri::TextureBarrierDesc*)resourceSnapshot.unique[i].userArg;
                state->before = state->after;
                state->after = resourceSnapshot.unique[i].state;
            }
        }
    }

    inline void InitCmdLine(cmdline::parser& cmdLine) override {
        cmdLine.add<int32_t>("dlssQuality", 'd', "DLSS quality: [-1: 4]", false, -1, cmdline::range(-1, 4));
        cmdLine.add("debugNRD", 0, "enable NRD validation");
    }

    inline void ReadCmdLine(cmdline::parser& cmdLine) override {
        m_DlssQuality = cmdLine.get<int32_t>("dlssQuality");
        m_DebugNRD = cmdLine.exist("debugNRD");
    }

    inline nrd::RelaxSettings GetDefaultRelaxSettings() const {
        nrd::RelaxSettings defaults = {};
        defaults.checkerboardMode = nrd::CheckerboardMode::OFF;
        defaults.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
        defaults.minMaterialForSpecular = MATERIAL_ID_METAL;
        defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        defaults.diffuseMaxAccumulatedFrameNum = m_RelaxSettings.diffuseMaxAccumulatedFrameNum;
        defaults.specularMaxAccumulatedFrameNum = m_RelaxSettings.specularMaxAccumulatedFrameNum;
        defaults.diffuseMaxFastAccumulatedFrameNum = m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum;
        defaults.specularMaxFastAccumulatedFrameNum = m_RelaxSettings.specularMaxFastAccumulatedFrameNum;

        // Helps to mitigate fireflies emphasized by DLSS
        // defaults.enableAntiFirefly = m_DlssQuality != -1 && IsDlssEnabled(); // TODO: currently doesn't help in this case, but makes the image darker

        return defaults;
    }

    inline nrd::ReblurSettings GetDefaultReblurSettings() const {
        nrd::ReblurSettings defaults = {};
        defaults.checkerboardMode = nrd::CheckerboardMode::OFF;
        defaults.minMaterialForDiffuse = MATERIAL_ID_DEFAULT;
        defaults.minMaterialForSpecular = MATERIAL_ID_METAL;
        defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
        defaults.maxAccumulatedFrameNum = m_ReblurSettings.maxAccumulatedFrameNum;
        defaults.maxFastAccumulatedFrameNum = m_ReblurSettings.maxFastAccumulatedFrameNum;
        defaults.maxStabilizedFrameNum = m_ReblurSettings.maxStabilizedFrameNum;

        // Helps to mitigate fireflies emphasized by DLSS
        defaults.enableAntiFirefly = m_DlssQuality != -1 && IsDlssEnabled();

        return defaults;
    }

    inline float3 GetSunDirection() const {
        float3 sunDirection;
        sunDirection.x = cos(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.y = sin(radians(m_Settings.sunAzimuth)) * cos(radians(m_Settings.sunElevation));
        sunDirection.z = sin(radians(m_Settings.sunElevation));

        return sunDirection;
    }

    bool Initialize(nri::GraphicsAPI graphicsAPI, bool) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    void LoadScene();
    void AddInnerGlassSurfaces();
    void GenerateAnimatedCubes();
    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipelineLayoutAndDescriptorPool();
    void CreatePipelines();
    void CreateAccelerationStructures();
    void CreateResources(nri::Format swapChainFormat);
    void CreateDescriptorSets();
    void CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, nri::Dim_t width, nri::Dim_t height, nri::Dim_t mipNum, nri::Dim_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state);
    void CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage);
    void UploadStaticData();
    void UpdateConstantBuffer(uint32_t frameIndex, float resetHistoryFactor);
    void RestoreBindings(nri::CommandBuffer& commandBuffer);
    void GatherInstanceData();
    uint32_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureBarrierDesc, MAX_TEXTURE_TRANSITIONS_NUM>& transitions);

private:
    // NRD
    nrd::Integration m_NRD = {};
    nrd::RelaxSettings m_RelaxSettings = {};
    nrd::ReblurSettings m_ReblurSettings = {};
    nrd::SigmaSettings m_SigmaSettings = {};
    nrd::ReferenceSettings m_ReferenceSettings = {};

    // NRI
    NRIInterface NRI = {};
    utils::Scene m_Scene;
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::Upscaler* m_DLSR = nullptr;
    nri::Upscaler* m_DLRR = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    std::array<nri::Upscaler*, 2> m_NIS = {};
    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureBarrierDesc> m_TextureStates;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_AccelerationStructures;
    std::vector<SwapChainTexture> m_SwapChainTextures;

    // Data
    std::vector<InstanceData> m_InstanceData;
    std::vector<nri::TopLevelInstance> m_WorldTlasData;
    std::vector<nri::TopLevelInstance> m_LightTlasData;
    std::vector<AnimatedInstance> m_AnimatedInstances;
    std::array<float, 256> m_FrameTimes = {};
    Settings m_Settings = {};
    Settings m_SettingsPrev = {};
    Settings m_SettingsDefault = {};
    const std::vector<uint32_t>* m_checkMeTests = nullptr;
    const std::vector<uint32_t>* m_improveMeTests = nullptr;
    float4 m_HairBaseColor = float4(0.1f, 0.1f, 0.1f, 1.0f);
    float3 m_PrevLocalPos = {};
    float2 m_HairBetas = float2(0.25f, 0.3f);
    uint2 m_RenderResolution = {};
    nri::BufferOffset m_WorldTlasDataLocation = {};
    nri::BufferOffset m_LightTlasDataLocation = {};
    uint32_t m_GlobalConstantBufferOffset = 0;
    uint32_t m_OpaqueObjectsNum = 0;
    uint32_t m_TransparentObjectsNum = 0;
    uint32_t m_EmissiveObjectsNum = 0;
    uint32_t m_ProxyInstancesNum = 0;
    uint32_t m_LastSelectedTest = uint32_t(-1);
    uint32_t m_TestNum = uint32_t(-1);
    int32_t m_DlssQuality = int32_t(-1);
    float m_UiWidth = 0.0f;
    float m_MinResolutionScale = 0.5f;
    float m_DofAperture = 0.0f;
    float m_DofFocalDistance = 1.0f;
    float m_SdrScale = 1.0f;
    bool m_ShowUi = true;
    bool m_ForceHistoryReset = false;
    bool m_Resolve = true;
    bool m_DebugNRD = false;
    bool m_ShowValidationOverlay = false;
    bool m_IsSrgb = false;
    bool m_GlassObjects = false;
    bool m_IsReloadShadersSucceeded = true;

#pragma region[ OmmSample specific ]
private: // OMM:
    void GenerateGeometry(utils::Scene& scene);
    void GeneratePlane(utils::Scene& scene, float3 origin, float3 axisX, float3 axisY, float2 size, uint32_t subdivision, uint32_t vertexOffset, float uvScaling);
    void PushVertex(utils::Scene& scene, float positionX, float positionY, float positionZ, float texCoordU, float texCoordV);
    void ComputePrimitiveNormal(utils::Scene& scene, uint32_t vertexOffset, uint32_t indexOffset);

    struct OmmNriContext;
    void InitAlphaTestedGeometry();

    void RebuildOmmGeometry();
    void RebuildOmmGeometryAsync(uint32_t const* frameId);
    void OmmGeometryUpdate(OmmNriContext& context, bool doBatching);

    void FillOmmBakerInputs();
    void FillOmmBlasBuildQueue(const OmmBatch& batch, std::vector<ommhelper::MaskedGeometryBuildDesc*>& outBuildQueue);

    void RunOmmSetupPass(OmmNriContext& context, ommhelper::OmmBakeGeometryDesc** queue, size_t count, OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void BakeOmmGpu(OmmNriContext& context, std::vector<ommhelper::OmmBakeGeometryDesc*>& batch);
    OmmGpuBakerPrebuildMemoryStats GetGpuBakerPrebuildMemoryStats(bool printStats);

    void CreateAndBindGpuBakerSatitcBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void CreateAndBindGpuBakerArrayDataBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void CreateAndBindGpuBakerReadbackBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats);

    inline uint64_t GetInstanceHash(uint32_t meshId, uint32_t materialId) {
        return uint64_t(meshId) << 32 | uint64_t(materialId);
    };

    inline std::string GetOmmCacheFilename() {
        return m_OmmCacheFolderName + std::string("/") + m_SceneName;
    };

    void InitializeOmmGeometryFromCache(const OmmBatch& batch, std::vector<ommhelper::OmmBakeGeometryDesc*>& outBakeQueue);
    void SaveMaskCache(const OmmBatch& batch);

    nri::AccelerationStructure* GetMaskedBlas(uint64_t insatanceMask);

    void ReleaseMaskedGeometry();
    void ReleaseBakingResources();

    void AppendOmmImguiSettings();

private:
    struct OmmNriContext {
        void Init(const NRIInterface& NRI, nri::Device* device, nri::QueueType type);
        void Destroy(const NRIInterface& NRI);

        nri::CommandAllocator* commandAllocator;
        nri::CommandBuffer* commandBuffer;
        nri::Queue* commandQueue;
        nri::Fence* fence;
        uint64_t fenceValue = 0;
    };

    ommhelper::OpacityMicroMapsHelper m_OmmHelper = {};

    // preprocessed alpha geometry from the scene:
    std::vector<AlphaTestedGeometry> m_OmmAlphaGeometry;
    std::vector<nri::Memory*> m_OmmAlphaGeometryMemories;
    std::vector<nri::Buffer*> m_OmmAlphaGeometryBuffers;

    // temporal resources for baking
    std::vector<uint8_t> m_OmmRawAlphaChannelForCpuBaker;

    nri::Buffer* m_OmmGpuOutputBuffers[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    nri::Buffer* m_OmmGpuReadbackBuffers[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    nri::Buffer* m_OmmGpuTransientBuffers[OMM_MAX_TRANSIENT_POOL_BUFFERS] = {};

    std::vector<nri::Buffer*> m_OmmCpuUploadBuffers;
    std::vector<nri::Memory*> m_OmmBakerAllocations;
    std::vector<nri::Memory*> m_OmmTmpAllocations;

    // misc
    OmmNriContext m_OmmGraphicsContext;
    OmmNriContext m_OmmComputeContext;

    struct OmmBlas {
        nri::AccelerationStructure* blas;
        //[!] VK Warning! VkMicromapExt wrapping is not supported yet. Use OmmHelper::DestroyMaskedGeometry instead of nri on release.
        nri::Buffer* ommArray;
    };

    std::map<uint64_t, OmmBlas> m_InstanceMaskToMaskedBlasData;
    std::vector<OmmBlas> m_MaskedBlasses;
    ommhelper::OmmBakeDesc m_OmmBakeDesc = {};
    std::string m_SceneName = "Scene";
    std::string m_OmmCacheFolderName = "_OmmCache";
    uint32_t m_OmmUpdateProgress = 0;
    bool m_EnableOmm = true;
    bool m_ShowFullSettings = false;
    bool m_IsOmmBakingActive = false;
    bool m_ShowOnlyAlphaTestedGeometry = false;
    bool m_EnableAsync = true;
    bool m_DisableOmmBlasBuild = false;

private:
    Profiler m_Profiler;
#pragma endregion
};


Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocator);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        for (uint32_t i = 0; i < m_Textures.size(); i++)
            NRI.DestroyTexture(m_Textures[i]);

        for (uint32_t i = 0; i < m_Buffers.size(); i++)
            NRI.DestroyBuffer(m_Buffers[i]);

        for (uint32_t i = 0; i < m_Descriptors.size(); i++)
            NRI.DestroyDescriptor(m_Descriptors[i]);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(m_Pipelines[i]);

        for (uint32_t i = 0; i < m_AccelerationStructures.size(); i++)
            NRI.DestroyAccelerationStructure(m_AccelerationStructures[i]);

        NRI.DestroyPipelineLayout(m_PipelineLayout);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
        NRI.DestroyFence(m_FrameFence);
    }

    if (NRI.HasUpscaler()) {
        NRI.DestroyUpscaler(m_NIS[0]);
        NRI.DestroyUpscaler(m_NIS[1]);
        NRI.DestroyUpscaler(m_DLSR);
        NRI.DestroyUpscaler(m_DLRR);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(m_Streamer);

    m_NRD.Destroy();

#pragma region[ OmmSample specific ]
    m_Profiler.Destroy();
    ReleaseMaskedGeometry();
    ReleaseBakingResources();
    m_OmmHelper.Destroy();
    m_OmmGraphicsContext.Destroy(NRI);
    m_OmmComputeContext.Destroy(NRI);
#pragma endregion

    DestroyImgui();

    nri::nriDestroyDevice(m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    Rng::Hash::Initialize(m_RngState, 106937, 69);

    // Adapters
    nri::AdapterDesc adapterDesc[4] = {};
    uint32_t adapterDescsNum = helper::GetCountOf(adapterDesc);
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(adapterDesc, adapterDescsNum));

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_ENABLE_COMMAND_BUFFER_EMULATION;
    deviceCreationDesc.disableD3D12EnhancedBarriers = D3D12_DISABLE_ENHANCED_BARRIERS;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &adapterDesc[std::min(m_AdapterIndex, adapterDescsNum - 1)];
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
#pragma region[ OmmSample specific ]
    const nri::QueueFamilyDesc queueFamilies[] = {
        {nullptr, 1, nri::QueueType::GRAPHICS},
        {nullptr, 1, nri::QueueType::COMPUTE},
    };
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = uint32_t(std::size(queueFamilies));
#pragma endregion
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::UpscalerInterface), (nri::UpscalerInterface*)&NRI));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Create streamer
        nri::StreamerDesc streamerDesc = {};
        streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::DEVICE_UPLOAD;
        streamerDesc.constantBufferSize = DYNAMIC_CONSTANT_BUFFER_SIZE;
        streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::DEVICE_UPLOAD;
        streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT};
        streamerDesc.queuedFrameNum = GetQueuedFrameNum();
        NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));
    }

    { // Create upscaler: NIS
        nri::UpscalerDesc upscalerDesc = {};
        upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
        upscalerDesc.type = nri::UpscalerType::NIS;

        upscalerDesc.flags = nri::UpscalerBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_NIS[0]));

        upscalerDesc.flags = nri::UpscalerBits::HDR;
        NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_NIS[1]));
    }

    // Create upscalers: DLSR and DLRR
    m_RenderResolution = GetOutputResolution();

    if (m_DlssQuality != -1) {
        const nri::UpscalerBits upscalerFlags = nri::UpscalerBits::DEPTH_INFINITE | nri::UpscalerBits::HDR;

        nri::UpscalerMode mode = nri::UpscalerMode::NATIVE;
        if (m_DlssQuality == 0)
            mode = nri::UpscalerMode::ULTRA_PERFORMANCE;
        else if (m_DlssQuality == 1)
            mode = nri::UpscalerMode::PERFORMANCE;
        else if (m_DlssQuality == 2)
            mode = nri::UpscalerMode::BALANCED;
        else if (m_DlssQuality == 3)
            mode = nri::UpscalerMode::QUALITY;

        if (NRI.IsUpscalerSupported(*m_Device, nri::UpscalerType::DLSR)) {
            nri::VideoMemoryInfo videoMemoryInfo1 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

            nri::UpscalerDesc upscalerDesc = {};
            upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
            upscalerDesc.type = upscalerType;
            upscalerDesc.mode = mode;
            upscalerDesc.flags = upscalerFlags;
            upscalerDesc.preset = USE_DLSS_TNN ? 10 : 0;
            NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_DLSR));

            nri::UpscalerProps upscalerProps = {};
            NRI.GetUpscalerProps(*m_DLSR, upscalerProps);

            float sx = float(upscalerProps.renderResolutionMin.w) / float(upscalerProps.renderResolution.w);
            float sy = float(upscalerProps.renderResolutionMin.h) / float(upscalerProps.renderResolution.h);

            m_RenderResolution = {upscalerProps.renderResolution.w, upscalerProps.renderResolution.h};
            m_MinResolutionScale = sy > sx ? sy : sx;

            nri::VideoMemoryInfo videoMemoryInfo2 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

            printf("Render resolution (%u, %u)\n", m_RenderResolution.x, m_RenderResolution.y);
            printf("DLSS-SR: allocated %.2f Mb\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));

            m_Settings.SR = true;
        }

        if (NRI.IsUpscalerSupported(*m_Device, nri::UpscalerType::DLRR)) {
            nri::VideoMemoryInfo videoMemoryInfo1 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

            nri::UpscalerDesc upscalerDesc = {};
            upscalerDesc.upscaleResolution = {(nri::Dim_t)GetOutputResolution().x, (nri::Dim_t)GetOutputResolution().y};
            upscalerDesc.type = nri::UpscalerType::DLRR;
            upscalerDesc.mode = mode;
            upscalerDesc.flags = upscalerFlags;
            NRI_ABORT_ON_FAILURE(NRI.CreateUpscaler(*m_Device, upscalerDesc, m_DLRR));

            nri::VideoMemoryInfo videoMemoryInfo2 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

            printf("DLSS-RR: allocated %.2f Mb\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));

            m_Settings.RR = true;
        }
    }

    // Initialize NRD: REBLUR, RELAX and SIGMA in one instance
    {
        const nrd::DenoiserDesc denoisersDescs[] = {
        // REBLUR
#if (NRD_MODE == SH)
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR_SH), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH},
#else
            {NRD_ID(REBLUR_DIFFUSE_SPECULAR), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR},
#endif

        // RELAX
#if (NRD_MODE == SH)
            {NRD_ID(RELAX_DIFFUSE_SPECULAR_SH), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH},
#else
            {NRD_ID(RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR},
#endif

        // SIGMA
            {NRD_ID(SIGMA_SHADOW), SIGMA_VARIANT},

            // REFERENCE
            {NRD_ID(REFERENCE), nrd::Denoiser::REFERENCE},
        };

        nrd::InstanceCreationDesc instanceCreationDesc = {};
        instanceCreationDesc.denoisers = denoisersDescs;
        instanceCreationDesc.denoisersNum = helper::GetCountOf(denoisersDescs);

        nrd::IntegrationCreationDesc desc = {};
        strcpy(desc.name, "NRD");
        desc.queuedFrameNum = GetQueuedFrameNum();
        desc.enableWholeLifetimeDescriptorCaching = true;
        desc.demoteFloat32to16 = false;
        desc.resourceWidth = (uint16_t)m_RenderResolution.x;
        desc.resourceHeight = (uint16_t)m_RenderResolution.y;
        desc.autoWaitForIdle = false;

        nri::VideoMemoryInfo videoMemoryInfo1 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

        if (m_NRD.Recreate(desc, instanceCreationDesc, m_Device) != nrd::Result::SUCCESS)
            return false;

        nri::VideoMemoryInfo videoMemoryInfo2 = {};
        NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

        printf("NRD: allocated %.2f Mb for REBLUR, RELAX, SIGMA and REFERENCE denoisers\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));
    }

    LoadScene();
#pragma region[ OmmSample specific ]
    for (size_t i = 0; i < m_Scene.instances.size(); ++i) {
        utils::Instance& instance = m_Scene.instances[i];
        utils::Material& material = m_Scene.materials[instance.materialIndex];
        instance.allowUpdate = material.IsAlphaOpaque() ? true : instance.allowUpdate;
    }
    GenerateGeometry(m_Scene);
#pragma endregion

    if (m_SceneFile.find("BistroInterior") != std::string::npos)
        AddInnerGlassSurfaces();

    GenerateAnimatedCubes();

    nri::Format swapChainFormat = CreateSwapChain();
    CreateCommandBuffers();
    CreatePipelineLayoutAndDescriptorPool();
    CreatePipelines();
    CreateAccelerationStructures();
    CreateResources(swapChainFormat);
    CreateDescriptorSets();

    UploadStaticData();

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);

#pragma region[ Omm Sample specific ]
    InitAlphaTestedGeometry();
    m_OmmHelper.Initialize(m_Device, m_DisableOmmBlasBuild);
    m_Profiler.Init(m_Device);
    m_OmmGraphicsContext.Init(NRI, m_Device, nri::QueueType::GRAPHICS);
    m_OmmComputeContext.Init(NRI, m_Device, nri::QueueType::COMPUTE);


    size_t sceneBeginNameOffset = m_SceneFile.find_last_of("/");
    sceneBeginNameOffset = sceneBeginNameOffset == std::string::npos ? 0 : ++sceneBeginNameOffset;
    size_t sceneEndNameOffset = m_SceneFile.find_last_of(".");
    sceneEndNameOffset = sceneEndNameOffset == std::string::npos ? m_SceneFile.length() : sceneEndNameOffset;
    m_SceneName = m_SceneFile.substr(sceneBeginNameOffset, sceneEndNameOffset - sceneBeginNameOffset);

    float3 cameraInitialPos = m_Scene.aabb.GetCenter();
    float3 lookAtPos = m_Scene.aabb.vMin;
    if (m_SceneFile.find("BistroExterior") != std::string::npos) {
        cameraInitialPos = float3(49.545f, -38.352f, 6.916f);
        float3 realLookAtPos = float3(41.304f, -26.487f, 4.805f);
        float3 hackedDir = realLookAtPos - cameraInitialPos;
        hackedDir = float3(hackedDir.y, -hackedDir.x, hackedDir.z);
        lookAtPos = cameraInitialPos + hackedDir;
    }
    m_Camera.Initialize(cameraInitialPos, lookAtPos, CAMERA_RELATIVE);

    if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12) {
        std::string windowTitle = std::string(glfwGetWindowTitle(m_Window));
#if DXR_OMM
        windowTitle += " [DXR 1.2]";
#else
        windowTitle += " [NVAPI]";
#endif
        glfwSetWindowTitle(m_Window, windowTitle.c_str());
    }
#pragma endregion

    m_Scene.UnloadGeometryData();

    m_SettingsDefault = m_Settings;
    m_ShowValidationOverlay = m_DebugNRD;

    nri::VideoMemoryInfo videoMemoryInfo = {};
    NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo);
    printf("Allocated %.2f Mb\n", videoMemoryInfo.usageSize / (1024.0f * 1024.0f));

    return InitImgui(*m_Device);
}

void Sample::OmmNriContext::Init(const NRIInterface& NRI, nri::Device* device, nri::QueueType type) {
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*device, type, 0, commandQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*commandQueue, commandAllocator));
    NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*commandAllocator, commandBuffer));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*device, 0, fence));
}

void Sample::OmmNriContext::Destroy(const NRIInterface& NRI) {
    NRI.DestroyFence(fence);
    NRI.DestroyCommandBuffer(commandBuffer);
    NRI.DestroyCommandAllocator(commandAllocator);
}

void BindBuffersToMemory(NRIInterface& nri, nri::Device* device, nri::Buffer** buffers, size_t count, std::vector<nri::Memory*>& memories, nri::MemoryLocation location) {
    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.buffers = buffers;
    resourceGroupDesc.bufferNum = (uint32_t)count;
    resourceGroupDesc.memoryLocation = location;
    size_t allocationOffset = memories.size();
    memories.resize(allocationOffset + nri.CalculateAllocationNumber(*device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(nri.AllocateAndBindMemory(*device, resourceGroupDesc, memories.data() + allocationOffset));
}

nri::AccelerationStructure* Sample::GetMaskedBlas(uint64_t insatanceMask) {
    const auto& it = m_InstanceMaskToMaskedBlasData.find(insatanceMask);
    if (it != m_InstanceMaskToMaskedBlasData.end())
        return it->second.blas;
    return nullptr;
}

std::vector<uint32_t> FilterOutAlphaTestedGeometry(const utils::Scene& scene) { // Filter out alphaOpaque geometry by mesh and material IDs
    std::vector<uint32_t> result;
    std::set<uint64_t> processedCombinations;
    for (uint32_t instaceId = 0; instaceId < (uint32_t)scene.instances.size(); ++instaceId) {
        const utils::Instance& instance = scene.instances[instaceId];
        const utils::Material& material = scene.materials[instance.materialIndex];
        if (material.IsAlphaOpaque()) {
            uint64_t mask = uint64_t(instance.meshInstanceIndex) << 32 | uint64_t(instance.materialIndex);
            size_t currentCount = processedCombinations.size();
            processedCombinations.insert(mask);
            bool isDuplicate = processedCombinations.size() == currentCount;
            if (isDuplicate == false)
                result.push_back(instaceId);
        }
    }
    return result;
}

void Sample::InitAlphaTestedGeometry() {
    printf("[OMM] Initializing Alpha Tested Geometry\n");
    std::vector<uint32_t> alphaInstances = FilterOutAlphaTestedGeometry(m_Scene);

    if (alphaInstances.empty())
        return;

    m_OmmAlphaGeometry.resize(alphaInstances.size());

    size_t positionBufferSize = 0;
    size_t indexBufferSize = 0;
    size_t uvBufferSize = 0;

    for (size_t i = 0; i < alphaInstances.size(); ++i) { // Calculate buffer sizes
        const utils::Instance& instance = m_Scene.instances[alphaInstances[i]];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshInstanceIndex];

        positionBufferSize += helper::Align(mesh.vertexNum * sizeof(float3), 256);
        indexBufferSize += helper::Align(mesh.indexNum * sizeof(utils::Index), 256);
        uvBufferSize += helper::Align(mesh.vertexNum * sizeof(float2), 256);
    }

    m_OmmAlphaGeometryBuffers.reserve(3);
    nri::Buffer*& positionBuffer = m_OmmAlphaGeometryBuffers.emplace_back();
    nri::Buffer*& indexBuffer = m_OmmAlphaGeometryBuffers.emplace_back();
    nri::Buffer*& uvBuffer = m_OmmAlphaGeometryBuffers.emplace_back();

    { // Cteate buffers
        nri::BufferDesc bufferDesc = {};
        bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT;

        bufferDesc.size = positionBufferSize;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, positionBuffer));

        bufferDesc.size = indexBufferSize;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, indexBuffer));

        // uv buffer is used in OMM baking as a raw read buffer. For compatibility with Vulkan this buffer is required to be structured
        bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE;
        bufferDesc.size = uvBufferSize;
        bufferDesc.structureStride = sizeof(uint32_t);
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, uvBuffer));
    }

    // raw data for uploading to gpu
    std::vector<uint8_t> positions;
    std::vector<uint8_t> uvs;
    std::vector<uint8_t> indices;

    uint32_t storageAlignment = NRI.GetDeviceDesc(*m_Device).memoryAlignment.bufferShaderResourceOffset;
    uint32_t bufferAlignment = storageAlignment;

    nri::Texture** materialTextures = m_Textures.data() + (size_t)Texture::MaterialTextures;
    for (size_t i = 0; i < alphaInstances.size(); ++i) {
        const utils::Instance& instance = m_Scene.instances[alphaInstances[i]];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshInstanceIndex];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        geometry.meshIndex = instance.meshInstanceIndex;
        geometry.materialIndex = instance.materialIndex;

        geometry.alphaTexture = materialTextures[material.baseColorTexIndex];
        geometry.utilsTexture = m_Scene.textures[material.baseColorTexIndex];

        size_t uvDataSize = mesh.vertexNum * sizeof(float2);
        geometry.uvData.resize(uvDataSize);

        size_t positionDataSize = mesh.vertexNum * sizeof(float3);
        geometry.positions = positionBuffer;
        geometry.positionOffset = positions.size();
        geometry.positionBufferSize = positionBufferSize;
        positions.resize(geometry.positionOffset + helper::Align(positionDataSize, bufferAlignment));

        for (uint32_t y = 0; y < mesh.vertexNum; ++y) {
            uint32_t offset = mesh.vertexOffset + y;
            memcpy(geometry.uvData.data() + y * sizeof(float2), m_Scene.unpackedVertices[offset].uv, sizeof(float2));

            float3 position = {
                m_Scene.unpackedVertices[offset].pos[0],
                m_Scene.unpackedVertices[offset].pos[1],
                m_Scene.unpackedVertices[offset].pos[2],
            };
            const size_t positionStride = sizeof(float3);
            void* dst = positions.data() + geometry.positionOffset + (y * positionStride);
            memcpy(dst, &position, positionStride);
        }

        size_t indexDataSize = mesh.indexNum * sizeof(utils::Index);
        geometry.indexData.resize(indexDataSize);
        memcpy(geometry.indexData.data(), m_Scene.indices.data() + mesh.indexOffset, indexDataSize);

        geometry.indices = indexBuffer;
        geometry.indexOffset = indices.size();
        geometry.indexBufferSize = indexBufferSize;
        indices.resize(geometry.indexOffset + helper::Align(indexDataSize, bufferAlignment));
        memcpy(indices.data() + geometry.indexOffset, m_Scene.indices.data() + mesh.indexOffset, indexDataSize);

        geometry.uvs = uvBuffer;
        geometry.uvOffset = uvs.size();
        geometry.uvBufferSize = uvBufferSize;
        uvs.resize(geometry.uvOffset + helper::Align(uvDataSize, storageAlignment));
        memcpy(uvs.data() + geometry.uvOffset, geometry.uvData.data(), uvDataSize);
    }

    { // Bind memories
        BindBuffersToMemory(NRI, m_Device, m_OmmAlphaGeometryBuffers.data(), m_OmmAlphaGeometryBuffers.size(), m_OmmAlphaGeometryMemories, nri::MemoryLocation::DEVICE);
    }

    std::vector<nri::BufferUploadDesc> uploadDescs;
    {
        nri::BufferUploadDesc desc = {};
        desc.after.access = nri::AccessBits::SHADER_RESOURCE;
        desc.buffer = positionBuffer;
        desc.data = positions.data();
        uploadDescs.push_back(desc);

        desc.buffer = uvBuffer;
        desc.data = uvs.data();
        uploadDescs.push_back(desc);

        desc.buffer = indexBuffer;
        desc.data = indices.data();
        uploadDescs.push_back(desc);
    }
    NRI.UploadData(*m_GraphicsQueue, nullptr, 0, uploadDescs.data(), (uint32_t)uploadDescs.size());
}

void PreprocessAlphaTexture(detexTexture* texture, std::vector<uint8_t>& outAlphaChannel) {
    uint8_t* pixels = texture->data;
    std::vector<uint8_t> decompressedImage;
    uint32_t format = texture->format;
    { // Hack detex to decompress texture as BC1A to get alpha data
        uint32_t originalFormat = texture->format;
        if (originalFormat == DETEX_TEXTURE_FORMAT_BC1)
            texture->format = DETEX_TEXTURE_FORMAT_BC1A;

        if (detexFormatIsCompressed(texture->format)) {
            uint32_t size = uint32_t(texture->width) * uint32_t(texture->height) * detexGetPixelSize(DETEX_PIXEL_FORMAT_RGBA8);
            decompressedImage.resize(size);
            detexDecompressTextureLinear(texture, &decompressedImage[0], DETEX_PIXEL_FORMAT_RGBA8);
            pixels = &decompressedImage[0];
            format = DETEX_PIXEL_FORMAT_RGBA8;
        }
        texture->format = originalFormat;
    }

    uint32_t pixelSize = detexGetPixelSize(format);
    uint32_t pixelCount = texture->width * texture->height;
    outAlphaChannel.reserve(pixelCount);

    for (uint32_t i = 0; i < pixelCount; ++i) {
        uint32_t offset = i * pixelSize;
        uint32_t alphaValue;
        if (pixelSize == 4) {
            uint32_t pixel = *(uint32_t*)(pixels + offset);
            alphaValue = detexPixel32GetA8(pixel);
        } else {
            uint64_t pixel = *(uint64_t*)(pixels + offset);
            alphaValue = (uint32_t)detexPixel64GetA16(pixel);
        }
        outAlphaChannel.push_back(uint8_t(alphaValue));
    }
}

inline bool AreBakerOutputsOnGPU(const ommhelper::OmmBakeGeometryDesc& instance) {
    bool result = true;
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i)
        result &= bool(instance.gpuBuffers[i].dataSize);
    return result;
}

void Sample::FillOmmBakerInputs() {
    std::map<uint64_t, size_t> materialMaskToTextureDataOffset;
    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::CPU) { // Decompress textures and store alpha channel in a separate buffer for cpu baker
        std::set<uint32_t> uniqueMaterialIds;
        std::vector<uint8_t> workVector;
        for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i) { // Sort out unique textures to avoid resource duplication
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
            ommhelper::InputTexture& bakerTexure = geometry.bakeDesc.texture;
            uint32_t materialId = geometry.materialIndex;

            const utils::Material& material = m_Scene.materials[materialId];
            utils::Texture* utilsTexture = m_Scene.textures[material.baseColorTexIndex];

            uint32_t minMip = utilsTexture->GetMipNum() - 1;
            uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;
            uint32_t remainingMips = minMip - textureMipOffset + 1;
            uint32_t mipRange = m_OmmBakeDesc.mipCount > remainingMips ? remainingMips : m_OmmBakeDesc.mipCount;

            bakerTexure.mipOffset = textureMipOffset;
            bakerTexure.mipNum = mipRange;

            size_t uniqueMaterialsNum = uniqueMaterialIds.size();
            uniqueMaterialIds.insert(materialId);
            if (uniqueMaterialsNum == uniqueMaterialIds.size())
                continue; // duplication

            for (uint32_t mip = 0; mip < mipRange; ++mip) {
                uint32_t mipId = textureMipOffset + mip;
                detexTexture* texture = (detexTexture*)utilsTexture->mips[mipId];

                PreprocessAlphaTexture(texture, workVector);

                size_t rawBufferOffset = m_OmmRawAlphaChannelForCpuBaker.size();
                m_OmmRawAlphaChannelForCpuBaker.insert(m_OmmRawAlphaChannelForCpuBaker.end(), workVector.begin(), workVector.end());
                materialMaskToTextureDataOffset.insert(std::make_pair(uint64_t(materialId) << 32 | uint64_t(mipId), rawBufferOffset));
                workVector.clear();
            }
        }
    }

    for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i) { // Fill baking queue desc
        bool isGpuBaker = m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU;

        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& ommDesc = geometry.bakeDesc;
        const utils::Mesh& mesh = m_Scene.meshes[geometry.meshIndex];
        const utils::Material& material = m_Scene.materials[geometry.materialIndex];
        nri::Texture* texture = geometry.alphaTexture;

        ommhelper::InputTexture bakerTexture = ommDesc.texture;
        utils::Texture* utilsTexture = m_Scene.textures[material.baseColorTexIndex];
        if (isGpuBaker) {
            ommDesc.indices.nriBufferOrPtr.buffer = geometry.indices;
            ommDesc.uvs.nriBufferOrPtr.buffer = geometry.uvs;
            uint32_t minMip = utilsTexture->GetMipNum() - 1;
            uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;
            ommDesc.texture.mipOffset = textureMipOffset;
            ommDesc.texture.mipNum = 1; // gpu baker currently doesn't support multiple mips
            ommhelper::MipDesc& mipDesc = ommDesc.texture.mips[0];
            mipDesc.nriTextureOrPtr.texture = texture;
            mipDesc.width = reinterpret_cast<detexTexture*>(utilsTexture->mips[bakerTexture.mipOffset])->width;
            ;
            mipDesc.height = reinterpret_cast<detexTexture*>(utilsTexture->mips[bakerTexture.mipOffset])->height;
            ;
        } else {
            ommDesc.indices.nriBufferOrPtr.ptr = (void*)geometry.indexData.data();
            ommDesc.uvs.nriBufferOrPtr.ptr = (void*)geometry.uvData.data();

            for (uint32_t mip = 0; mip < bakerTexture.mipNum; ++mip) {
                uint32_t mipId = bakerTexture.mipOffset + mip;
                uint64_t materialMask = uint64_t(geometry.materialIndex) << 32 | uint64_t(mipId);
                size_t texDataOffset = materialMaskToTextureDataOffset.find(materialMask)->second;

                ommhelper::MipDesc& mipDesc = ommDesc.texture.mips[mip];
                mipDesc.nriTextureOrPtr.ptr = (void*)(m_OmmRawAlphaChannelForCpuBaker.data() + texDataOffset);
                mipDesc.width = reinterpret_cast<detexTexture*>(utilsTexture->mips[mipId])->width;
                mipDesc.height = reinterpret_cast<detexTexture*>(utilsTexture->mips[mipId])->height;
            }
        }

        ommDesc.indices.numElements = mesh.indexNum;
        ommDesc.indices.stride = sizeof(utils::Index);
        ommDesc.indices.format = nri::Format::R32_UINT;
        ommDesc.indices.offset = geometry.indexOffset;
        ommDesc.indices.bufferSize = geometry.indexBufferSize;
        ommDesc.indices.offsetInStruct = 0;

        ommDesc.uvs.numElements = mesh.vertexNum;
        ommDesc.uvs.stride = sizeof(float2);
        ommDesc.uvs.format = nri::Format::RG32_SFLOAT;
        ommDesc.uvs.offset = geometry.uvOffset;
        ommDesc.uvs.bufferSize = geometry.uvBufferSize;
        ommDesc.uvs.offsetInStruct = 0;

        ommDesc.texture.format = isGpuBaker ? utilsTexture->format : nri::Format::R8_UNORM;
        ommDesc.texture.addressingMode = nri::AddressMode::REPEAT;
        ommDesc.texture.alphaChannelId = 3;
        ommDesc.alphaCutoff = 0.5f;
        ommDesc.borderAlpha = 0.0f;
        ommDesc.alphaMode = ommhelper::OmmAlphaMode::Test;
    }
}

void PrepareOmmUsageCountsBuffers(ommhelper::OpacityMicroMapsHelper& ommHelper, ommhelper::OmmBakeGeometryDesc& desc) { // Sanitize baker outputed usageCounts buffers to fit GAPI format
    uint32_t usageCountBuffers[] = {(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram, (uint32_t)ommhelper::OmmDataLayout::IndexHistogram};

    for (size_t i = 0; i < helper::GetCountOf(usageCountBuffers); ++i) {
        std::vector<uint8_t> buffer = desc.outData[usageCountBuffers[i]];
        size_t convertedCountsSize = 0;
        ommHelper.ConvertUsageCountsToApiFormat(nullptr, convertedCountsSize, buffer.data(), buffer.size());
        desc.outData[usageCountBuffers[i]].resize(convertedCountsSize);
        ommHelper.ConvertUsageCountsToApiFormat(desc.outData[usageCountBuffers[i]].data(), convertedCountsSize, buffer.data(), buffer.size());
    }
}

void PrepareCpuBuilderInputs(NRIInterface& NRI, const OmmBatch& batch, std::vector<AlphaTestedGeometry>& geometries) { // Copy raw mask data to the upload heaps to use during micromap and blas build
    for (size_t i = batch.offset; i < batch.offset + batch.count; ++i) {
        AlphaTestedGeometry& geometry = geometries[i];
        const ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        if (bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram].empty())
            continue;

        ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++y) {
            nri::Buffer* buffer = buildDesc.inputs.buffers[y].buffer;
            uint64_t mapSize = (uint64_t)bakeResult.outData[y].size();
            void* map = NRI.MapBuffer(*buffer, 0, mapSize);
            memcpy(map, bakeResult.outData[y].data(), bakeResult.outData[y].size());
            NRI.UnmapBuffer(*buildDesc.inputs.buffers[y].buffer);
        }
    }
}

void Sample::FillOmmBlasBuildQueue(const OmmBatch& batch, std::vector<ommhelper::MaskedGeometryBuildDesc*>& outBuildQueue) {
    outBuildQueue.clear();
    outBuildQueue.reserve(batch.count);

    size_t uploadBufferOffset = m_OmmCpuUploadBuffers.size();
    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
        const utils::Mesh& mesh = m_Scene.meshes[geometry.meshIndex];

        ommhelper::InputBuffer& vertices = buildDesc.inputs.vertices;
        vertices.nriBufferOrPtr.buffer = geometry.positions;
        vertices.format = geometry.vertexFormat;
        vertices.stride = sizeof(float3);
        vertices.numElements = mesh.vertexNum;
        vertices.offset = geometry.positionOffset;
        vertices.bufferSize = geometry.positionBufferSize;
        vertices.offsetInStruct = 0;

        ommhelper::InputBuffer& indices = buildDesc.inputs.indices;
        indices = bakeResult.indices;
        indices.nriBufferOrPtr.buffer = geometry.indices;

        if (bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram].empty())
            continue;

        buildDesc.inputs.ommIndexFormat = bakeResult.outOmmIndexFormat;
        buildDesc.inputs.ommIndexStride = bakeResult.outOmmIndexStride;

        PrepareOmmUsageCountsBuffers(m_OmmHelper, bakeResult);

        if (AreBakerOutputsOnGPU(bakeResult)) {
            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++j)
                buildDesc.inputs.buffers[j] = bakeResult.gpuBuffers[j];
        } else { // Create upload buffers to store baker output during ommArray/blas creation
            nri::BufferDesc bufferDesc = {};
            bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE;

            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++j) {
                bufferDesc.size = bakeResult.outData[j].size();
                buildDesc.inputs.buffers[j].dataSize = bufferDesc.size;
                buildDesc.inputs.buffers[j].bufferSize = bufferDesc.size;
                NRI.CreateBuffer(*m_Device, bufferDesc, buildDesc.inputs.buffers[j].buffer);
                m_OmmCpuUploadBuffers.push_back(buildDesc.inputs.buffers[j].buffer);
            }
        }

        buildDesc.inputs.descArrayHistogram = bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram].data();
        buildDesc.inputs.descArrayHistogramNum = bakeResult.outDescArrayHistogramCount;

        buildDesc.inputs.indexHistogram = bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram].data();
        buildDesc.inputs.indexHistogramNum = bakeResult.outIndexHistogramCount;
        outBuildQueue.push_back(&buildDesc);
    }

    if (m_OmmCpuUploadBuffers.empty() == false) { // Bind cpu baker output memories
        size_t uploadBufferCount = m_OmmCpuUploadBuffers.size() - uploadBufferOffset;
        BindBuffersToMemory(NRI, m_Device, m_OmmCpuUploadBuffers.data() + uploadBufferOffset, uploadBufferCount, m_OmmTmpAllocations, nri::MemoryLocation::HOST_UPLOAD);
        PrepareCpuBuilderInputs(NRI, batch, m_OmmAlphaGeometry);
    }

    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id) { // Release raw cpu side data. In case of cpu baker it's in the upload heaps, in case of gpu it's already saved as cache
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        for (uint32_t k = 0; k < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++k) {
            bakeResult.outData[k].resize(0);
            bakeResult.outData[k].shrink_to_fit();
        }
    }
}

void CopyBatchToReadBackBuffer(const NRIInterface& NRI, nri::CommandBuffer* commandBuffer, ommhelper::OmmBakeGeometryDesc& firstInBatch, ommhelper::OmmBakeGeometryDesc& lastInBatch, uint32_t bufferId) {
    ommhelper::GpuBakerBuffer& firstResource = firstInBatch.gpuBuffers[bufferId];
    ommhelper::GpuBakerBuffer& lastResource = lastInBatch.gpuBuffers[bufferId];
    ommhelper::GpuBakerBuffer& firstReadback = firstInBatch.readBackBuffers[bufferId];

    nri::Buffer* src = firstResource.buffer;
    nri::Buffer* dst = firstReadback.buffer;
    size_t srcOffset = firstResource.offset;
    size_t dstOffset = firstReadback.offset;

    size_t size = (lastResource.offset + lastResource.dataSize) - firstResource.offset; // total size of baker output for the batch
    NRI.CmdCopyBuffer(*commandBuffer, *dst, dstOffset, *src, srcOffset, size);
}

void CopyFromReadBackBuffer(const NRIInterface& NRI, ommhelper::OmmBakeGeometryDesc& desc, size_t id) {
    ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[id];
    nri::Buffer* readback = resource.buffer;

    size_t offset = resource.offset;
    size_t size = resource.dataSize;
    std::vector<uint8_t>& data = desc.outData[id];
    data.resize(size);

    void* map = NRI.MapBuffer(*readback, offset, size);
    memcpy(data.data(), map, size);

    ZeroMemory(map, size);
    NRI.UnmapBuffer(*readback);
}

OmmGpuBakerPrebuildMemoryStats Sample::GetGpuBakerPrebuildMemoryStats(bool printStats) {
    OmmGpuBakerPrebuildMemoryStats result = {};
    uint32_t sizeAlignment = NRI.GetDeviceDesc(*m_Device).memoryAlignment.micromapOffset;
    for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& instance = geometry.bakeDesc;
        ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& gpuBakerPreBuildInfo = instance.gpuBakerPreBuildInfo;

        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y) {
            gpuBakerPreBuildInfo.dataSizes[y] = helper::Align(gpuBakerPreBuildInfo.dataSizes[y], sizeAlignment);
            result.outputTotalSizes[y] += gpuBakerPreBuildInfo.dataSizes[y];
            result.outputMaxSizes[y] = std::max<size_t>(gpuBakerPreBuildInfo.dataSizes[y], result.outputMaxSizes[y]);
            result.total += gpuBakerPreBuildInfo.dataSizes[y];
        }

        for (size_t y = 0; y < OMM_MAX_TRANSIENT_POOL_BUFFERS; ++y) {
            gpuBakerPreBuildInfo.transientBufferSizes[y] = helper::Align(gpuBakerPreBuildInfo.transientBufferSizes[y], sizeAlignment);
            result.maxTransientBufferSizes[y] = std::max(result.maxTransientBufferSizes[y], gpuBakerPreBuildInfo.transientBufferSizes[y]);
        }
    }

    auto toBytes = [](size_t sizeInMb) -> size_t { return sizeInMb * 1024 * 1024; };
    const size_t defaultSizes[] = {toBytes(64), toBytes(5), toBytes(5), toBytes(5), toBytes(5), 1024};

    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU && printStats) {
        uint64_t totalPrimitiveNum = 0;
        uint64_t maxPrimitiveNum = 0;
        for (auto& geomtry : m_OmmAlphaGeometry) {
            uint64_t numPrimitives = geomtry.bakeDesc.indices.numElements / 3;
            totalPrimitiveNum += numPrimitives;
            maxPrimitiveNum = std::max<uint64_t>(maxPrimitiveNum, numPrimitives);
        }

        auto toMb = [](size_t sizeInBytes) -> double { return double(sizeInBytes) / 1024.0 / 1024.0; };
        printf("\n[OMM][GPU] PreBake Stats:\n");
        printf("Mask Format: [%s]\n", m_OmmBakeDesc.format == ommhelper::OmmFormats::OC1_2_STATE ? "OC1_2_STATE" : "OC1_4_STATE");
        printf("Subdivision Level: [%lu]\n", m_OmmBakeDesc.subdivisionLevel);
        printf("Mip Bias: [%lu]\n", m_OmmBakeDesc.mipBias);
        printf("Num Geometries: [%llu]\n", m_OmmAlphaGeometry.size());
        printf("Num Primitives: Max:[%llu],  Total:[%llu]\n", maxPrimitiveNum, totalPrimitiveNum);
        printf("Baker output memeory requested(mb): (total)%.3f\n", toMb(result.total));
        printf("Total ArrayDataSize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::ArrayData]));
        printf("Total DescArraySize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::DescArray]));
        printf("Total IndicesSize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::Indices]));
    }
    return result;
}

std::vector<OmmBatch> GetGpuBakerBatches(const std::vector<AlphaTestedGeometry>& geometries, const OmmGpuBakerPrebuildMemoryStats& memoryStats, const size_t batchSize) {
    const size_t batchMaxSize = batchSize > geometries.size() ? geometries.size() : batchSize;
    std::vector<OmmBatch> batches(1);
    size_t accumulation[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t i = 0; i < geometries.size(); ++i) {
        const AlphaTestedGeometry& geometry = geometries[i];
        const ommhelper::OmmBakeGeometryDesc& bakeDesc = geometry.bakeDesc;
        const ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& info = bakeDesc.gpuBakerPreBuildInfo;

        bool isAnyOverLimit = false;
        size_t nextSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y) {
            nextSizes[y] = accumulation[y] + info.dataSizes[y];
            isAnyOverLimit |= nextSizes[y] > memoryStats.outputMaxSizes[y];
        }

        if (isAnyOverLimit) {
            batches.push_back({i, 1});
            for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
                accumulation[y] = info.dataSizes[y];
            continue;
        }

        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
            accumulation[y] = nextSizes[y];

        ++batches.back().count;
        if (batches.back().count >= batchMaxSize) {
            if (i + 1 < geometries.size()) {
                batches.push_back({i + 1, 0});
                for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
                    accumulation[y] = 0;
                continue;
            }
        }
    }
    return batches;
}

void Sample::CreateAndBindGpuBakerReadbackBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats) { // for caching gpu produced omm_sdk output
    size_t dataTypeBegin = (size_t)ommhelper::OmmDataLayout::ArrayData;
    size_t dataTypeEnd = (size_t)ommhelper::OmmDataLayout::DescArrayHistogram;
    size_t micromapAlignment = NRI.GetDeviceDesc(*m_Device).memoryAlignment.micromapOffset;
    {
        for (size_t i = dataTypeBegin; i < dataTypeEnd; ++i) {
            nri::BufferDesc bufferDesc = {};
            bufferDesc.structureStride = sizeof(uint32_t);
            size_t s = memoryStats.outputTotalSizes[i];
            size_t a = micromapAlignment;
            bufferDesc.size = ((s + a - 1) / a) * a;
            // bufferDesc.size = memoryStats.outputTotalSizes[i];
            bufferDesc.usage = nri::BufferUsageBits::NONE;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuReadbackBuffers[i]));
        }
        BindBuffersToMemory(NRI, m_Device, &m_OmmGpuReadbackBuffers[dataTypeBegin], dataTypeEnd - dataTypeBegin, m_OmmBakerAllocations, nri::MemoryLocation::HOST_READBACK);
    }

    { // bind baker instances to the buffer
        size_t perDataTypeOffsets[(size_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
        for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id) {
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
            ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
            for (size_t i = dataTypeBegin; i < dataTypeEnd; ++i) {
                ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[i];
                size_t& offset = perDataTypeOffsets[i];

                resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[i];
                resource.buffer = m_OmmGpuReadbackBuffers[i];
                resource.bufferSize = memoryStats.outputTotalSizes[i];
                resource.offset = offset;
                offset += resource.dataSize;
            }
        }
    }
}

void Sample::CreateAndBindGpuBakerArrayDataBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats) { // in case of using setup pass of OMM-SDK, array data buffer allocation must be done separately
    const uint32_t arrayDataId = (uint32_t)ommhelper::OmmDataLayout::ArrayData;
    const size_t ommAlignment = NRI.GetDeviceDesc(*m_Device).memoryAlignment.micromapOffset;
    nri::BufferDesc bufferDesc = {};
    bufferDesc.structureStride = sizeof(uint32_t);
    bufferDesc.size = memoryStats.outputTotalSizes[arrayDataId];
    bufferDesc.size = helper::Align(bufferDesc.size, ommAlignment);
    bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuOutputBuffers[arrayDataId]));
    BindBuffersToMemory(NRI, m_Device, &m_OmmGpuOutputBuffers[arrayDataId], 1, m_OmmBakerAllocations, nri::MemoryLocation::DEVICE);

    size_t offset = 0;
    for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
        ommhelper::GpuBakerBuffer& resource = desc.gpuBuffers[arrayDataId];

        resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[arrayDataId];
        resource.buffer = m_OmmGpuOutputBuffers[arrayDataId];
        resource.bufferSize = memoryStats.outputTotalSizes[arrayDataId];
        resource.offset = offset;
        offset += desc.gpuBakerPreBuildInfo.dataSizes[arrayDataId];
    }
}

void Sample::CreateAndBindGpuBakerSatitcBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats) {
    const size_t postBakeReadbackDataBegin = (size_t)ommhelper::OmmDataLayout::DescArrayHistogram;
    const size_t staticDataBegin = (size_t)ommhelper::OmmDataLayout::DescArray;
    const size_t buffersEnd = (size_t)ommhelper::OmmDataLayout::GpuOutputNum;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.structureStride = sizeof(uint32_t);

    std::vector<nri::Buffer*> gpuBuffers;
    std::vector<nri::Buffer*> readbackBuffers;
    for (size_t i = staticDataBegin; i < buffersEnd; ++i) {
        bufferDesc.size = memoryStats.outputTotalSizes[i];
        bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuOutputBuffers[i]));
        gpuBuffers.push_back(m_OmmGpuOutputBuffers[i]);
    }

    for (size_t i = 0; i < OMM_MAX_TRANSIENT_POOL_BUFFERS; ++i) {
        bufferDesc.size = memoryStats.maxTransientBufferSizes[i];
        if (bufferDesc.size) {
            bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ARGUMENT_BUFFER;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuTransientBuffers[i]));
            gpuBuffers.push_back(m_OmmGpuTransientBuffers[i]);
        }
    }

    for (size_t i = postBakeReadbackDataBegin; i < buffersEnd; ++i) {
        bufferDesc.size = memoryStats.outputTotalSizes[i];
        bufferDesc.usage = nri::BufferUsageBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuReadbackBuffers[i]));
        readbackBuffers.push_back(m_OmmGpuReadbackBuffers[i]);
    }

    { // Bind memories
        BindBuffersToMemory(NRI, m_Device, gpuBuffers.data(), gpuBuffers.size(), m_OmmBakerAllocations, nri::MemoryLocation::DEVICE);
        BindBuffersToMemory(NRI, m_Device, readbackBuffers.data(), readbackBuffers.size(), m_OmmBakerAllocations, nri::MemoryLocation::HOST_READBACK);
    }

    size_t gpuOffsetsPerType[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    size_t readBackOffsetsPerType[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
        for (uint32_t j = staticDataBegin; j < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++j) {
            ommhelper::GpuBakerBuffer& resource = desc.gpuBuffers[j];
            size_t& offset = gpuOffsetsPerType[j];

            desc.gpuBuffers[j].dataSize = desc.gpuBakerPreBuildInfo.dataSizes[j];
            resource.buffer = m_OmmGpuOutputBuffers[j];
            resource.bufferSize = memoryStats.outputTotalSizes[j];
            resource.offset = offset;
            offset += desc.gpuBakerPreBuildInfo.dataSizes[j];
        }

        for (uint32_t j = postBakeReadbackDataBegin; j < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++j) {
            ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[j];
            size_t& offset = readBackOffsetsPerType[j];

            resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[j];
            resource.buffer = m_OmmGpuReadbackBuffers[j];
            resource.bufferSize = memoryStats.outputTotalSizes[j];
            resource.offset = offset;
            offset += resource.dataSize;
        }

        for (size_t j = 0; j < OMM_MAX_TRANSIENT_POOL_BUFFERS; ++j) {
            desc.transientBuffers[j].buffer = m_OmmGpuTransientBuffers[j];
            desc.transientBuffers[j].bufferSize = memoryStats.maxTransientBufferSizes[j];
            desc.transientBuffers[j].dataSize = memoryStats.maxTransientBufferSizes[j];
            desc.transientBuffers[j].offset = 0;
        }
    }
}

void Sample::SaveMaskCache(const OmmBatch& batch) {
    std::string cacheFileName = GetOmmCacheFilename();
    ommhelper::OmmCaching::CreateFolder(m_OmmCacheFolderName.c_str());
    uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);

    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResults = geometry.bakeDesc;
        uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);

        bool isDataValid = true;
        ommhelper::OmmCaching::OmmData data;
        for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i) {
            data.data[i] = bakeResults.outData[i].data();
            data.sizes[i] = bakeResults.outData[i].size();
            isDataValid &= data.sizes[i] > 0;
        }
        if (isDataValid)
            ommhelper::OmmCaching::SaveMasksToDisc(cacheFileName.c_str(), data, stateMask, hash, (uint16_t)bakeResults.outOmmIndexFormat);
    }
}

void Sample::InitializeOmmGeometryFromCache(const OmmBatch& batch, std::vector<ommhelper::OmmBakeGeometryDesc*>& outBakeQueue) { // Init geometry from cache. If cache not found add it to baking queue
    if (m_OmmBakeDesc.enableCache == false) {
        for (size_t i = batch.offset; i < batch.offset + batch.count; ++i)
            outBakeQueue.push_back(&m_OmmAlphaGeometry[i].bakeDesc);
        return;
    }

    printf("Read cache. ");
    uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);
    for (size_t i = batch.offset; i < batch.offset + batch.count; ++i) {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& instance = geometry.bakeDesc;

        uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);
        ommhelper::OmmCaching::OmmData data = {};
        if (ommhelper::OmmCaching::ReadMaskFromCache(GetOmmCacheFilename().c_str(), data, stateMask, hash, nullptr)) {
            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++j) {
                instance.outData[j].resize(data.sizes[j]);
                data.data[j] = instance.outData[j].data();
            }
            ommhelper::OmmCaching::ReadMaskFromCache(GetOmmCacheFilename().c_str(), data, stateMask, hash, (uint16_t*)&instance.outOmmIndexFormat);
            instance.outOmmIndexStride = instance.outOmmIndexFormat == nri::Format::R8_UINT ? sizeof(uint8_t) : instance.outOmmIndexFormat == nri::Format::R16_UINT ? sizeof(uint16_t)
                                                                                                                                                                    : sizeof(uint32_t);
            instance.outDescArrayHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram] / (uint64_t)sizeof(ommCpuOpacityMicromapUsageCount));
            instance.outIndexHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram] / (uint64_t)sizeof(ommCpuOpacityMicromapUsageCount));
        } else
            outBakeQueue.push_back(&instance);
    }
}

inline void SubmitQueueWorkAndWait(const NRIInterface& NRI, nri::CommandBuffer* commandBuffer, nri::Queue* queue, nri::Fence* fence, uint64_t& currentFenceValue) {
    nri::FenceSubmitDesc fenceSubmitDesc = {};
    fenceSubmitDesc.fence = fence;
    fenceSubmitDesc.stages = nri::StageBits::ALL;
    fenceSubmitDesc.value = ++currentFenceValue;

    nri::QueueSubmitDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;
    workSubmissionDesc.signalFences = &fenceSubmitDesc;
    workSubmissionDesc.signalFenceNum = 1;
    NRI.QueueSubmit(*queue, workSubmissionDesc);
    NRI.Wait(*fence, currentFenceValue);
}

void Sample::RunOmmSetupPass(OmmNriContext& context, ommhelper::OmmBakeGeometryDesc** queue, size_t count, OmmGpuBakerPrebuildMemoryStats& memoryStats) { // Run prepass to get correct size of omm array data buffer
    NRI.ResetCommandAllocator(*context.commandAllocator);
    NRI.BeginCommandBuffer(*context.commandBuffer, nullptr);
    {
        m_OmmHelper.BakeOpacityMicroMapsGpu(context.commandBuffer, queue, count, m_OmmBakeDesc, ommhelper::OmmGpuBakerPass::Setup);
        CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *queue[0], *queue[count - 1], (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
    }
    NRI.EndCommandBuffer(*context.commandBuffer);
    SubmitQueueWorkAndWait(NRI, context.commandBuffer, context.commandQueue, context.fence, context.fenceValue);
    m_OmmHelper.GpuPostBakeCleanUp();

    for (size_t i = 0; i < count; ++i) { // Get actual data sizes from postbuild info
        ommhelper::OmmBakeGeometryDesc& desc = *queue[i];
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
        ommGpuPostDispatchInfo postbildInfo = *(ommGpuPostDispatchInfo*)desc.outData[(uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo].data();
        desc.gpuBakerPreBuildInfo.dataSizes[(uint32_t)ommhelper::OmmDataLayout::ArrayData] = postbildInfo.outOmmArraySizeInBytes;
    }
    memoryStats = GetGpuBakerPrebuildMemoryStats(true);
}

void Sample::BakeOmmGpu(OmmNriContext& context, std::vector<ommhelper::OmmBakeGeometryDesc*>& batch) {
    NRI.ResetCommandAllocator(*context.commandAllocator);
    NRI.BeginCommandBuffer(*context.commandBuffer, nullptr);
    {
        m_OmmHelper.BakeOpacityMicroMapsGpu(context.commandBuffer, batch.data(), batch.size(), m_OmmBakeDesc, ommhelper::OmmGpuBakerPass::Bake);
        CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram);
        CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::IndexHistogram);
        CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
    }
    NRI.EndCommandBuffer(*context.commandBuffer);
    SubmitQueueWorkAndWait(NRI, context.commandBuffer, context.commandQueue, context.fence, context.fenceValue);
    m_OmmHelper.GpuPostBakeCleanUp();

    if (m_OmmBakeDesc.enableCache) {
        printf("Readback. ");
        NRI.ResetCommandAllocator(*context.commandAllocator);
        NRI.BeginCommandBuffer(*context.commandBuffer, nullptr);
        {
            for (size_t i = 0; i < batch.size(); ++i) { // Get actual data sizes from postbuild info
                ommhelper::OmmBakeGeometryDesc& desc = *batch[i];
                CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
                ommGpuPostDispatchInfo postbildInfo = *(ommGpuPostDispatchInfo*)desc.outData[(uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo].data();

                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
            }

            {
                CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::ArrayData);
                CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArray);
                CopyBatchToReadBackBuffer(NRI, context.commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::Indices);
            }
        }
        NRI.EndCommandBuffer(*context.commandBuffer);
        SubmitQueueWorkAndWait(NRI, context.commandBuffer, context.commandQueue, context.fence, context.fenceValue);
    }

    for (size_t i = 0; i < batch.size(); ++i) {
        ommhelper::OmmBakeGeometryDesc& desc = *batch[i];
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram);
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::IndexHistogram);

        if (m_OmmBakeDesc.enableCache) {
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::ArrayData);
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::DescArray);
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::Indices);
        }
    }
}

void Sample::OmmGeometryUpdate(OmmNriContext& context, bool doBatching) {
    ReleaseMaskedGeometry();
    FillOmmBakerInputs();
    OmmGpuBakerPrebuildMemoryStats memoryStats = {};
    std::vector<OmmBatch> batches = GetGpuBakerBatches(m_OmmAlphaGeometry, memoryStats, 1);

    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU) {
        std::vector<ommhelper::OmmBakeGeometryDesc*> queue;
        uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);

        for (size_t instanceId = 0; instanceId < m_OmmAlphaGeometry.size(); ++instanceId) { // skip prepass for instances with cache
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[instanceId];
            uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);
            if (ommhelper::OmmCaching::LookForCache(GetOmmCacheFilename().c_str(), stateMask, hash) && m_OmmBakeDesc.enableCache)
                continue;
            queue.push_back(&geometry.bakeDesc);
        }

        if (queue.empty() == false) { // perform setup pass
            m_OmmHelper.GetGpuBakerPrebuildInfo(queue.data(), queue.size(), m_OmmBakeDesc);
            memoryStats = GetGpuBakerPrebuildMemoryStats(false); // arrayData size calculation is conservative here

            CreateAndBindGpuBakerSatitcBuffers(memoryStats); // create buffers which sizes are correctly calculated in GetGpuBakerPrebuildInfo()
            {                                                // get actual arrayData buffer sizes. GetGpuBakerPrebuildInfo() returns conservative arrayData size estimation
                RunOmmSetupPass(context, queue.data(), queue.size(), memoryStats);
            }
            CreateAndBindGpuBakerArrayDataBuffer(memoryStats);

            if (m_OmmBakeDesc.enableCache)
                CreateAndBindGpuBakerReadbackBuffer(memoryStats);

            if (doBatching) {
                batches.clear();
                batches.push_back({0, m_OmmAlphaGeometry.size()});
            }
        }
    }

    for (size_t batchId = 0; batchId < batches.size(); ++batchId) {
        const OmmBatch& batch = batches[batchId];
        printf("\r%s\r[OMM] Batch [%llu / %llu]: ", std::string(100, ' ').c_str(), batchId + 1, batches.size());
        std::vector<ommhelper::OmmBakeGeometryDesc*> bakeQueue;
        InitializeOmmGeometryFromCache(batch, bakeQueue);

        if (!bakeQueue.empty()) {
            printf("Bake. ");
            if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU)
                BakeOmmGpu(context, bakeQueue);
            else
                m_OmmHelper.BakeOpacityMicroMapsCpu(bakeQueue.data(), bakeQueue.size(), m_OmmBakeDesc);

            if (m_OmmBakeDesc.enableCache) {
                printf("Save cache. ");
                SaveMaskCache(batch);
            }
        }

        if (m_DisableOmmBlasBuild == false) {
            printf("Build. ");

            std::vector<ommhelper::MaskedGeometryBuildDesc*> buildQueue = {};
            FillOmmBlasBuildQueue(batch, buildQueue);

            NRI.ResetCommandAllocator(*context.commandAllocator);
            NRI.BeginCommandBuffer(*context.commandBuffer, nullptr);
            {
                m_OmmHelper.BuildMaskedGeometry(buildQueue.data(), buildQueue.size(), context.commandBuffer);
            }
            NRI.EndCommandBuffer(*context.commandBuffer);
            SubmitQueueWorkAndWait(NRI, context.commandBuffer, context.commandQueue, context.fence, context.fenceValue);

            for (size_t id = batch.offset; id < batch.offset + batch.count; ++id) {
                AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
                ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
                if (!buildDesc.outputs.blas)
                    continue;

                uint64_t mask = GetInstanceHash(m_OmmAlphaGeometry[id].meshIndex, m_OmmAlphaGeometry[id].materialIndex);
                OmmBlas ommBlas = {buildDesc.outputs.blas, buildDesc.outputs.ommArray};
                m_InstanceMaskToMaskedBlasData.insert(std::make_pair(mask, ommBlas));
                m_MaskedBlasses.push_back({buildDesc.outputs.blas, buildDesc.outputs.ommArray});
            }
        }

        // Free cpu side memories with batch lifecycle
        for (auto& buffer : m_OmmCpuUploadBuffers)
            NRI.DestroyBuffer(buffer);
        m_OmmCpuUploadBuffers.resize(0);
        m_OmmCpuUploadBuffers.shrink_to_fit();

        for (auto& memory : m_OmmTmpAllocations)
            NRI.FreeMemory(memory);
        m_OmmTmpAllocations.resize(0);
        m_OmmTmpAllocations.shrink_to_fit();

        m_OmmUpdateProgress += (uint32_t)batch.count;
    }
    printf("\n");

    ReleaseBakingResources();
    m_OmmUpdateProgress = 0;
}

void Sample::RebuildOmmGeometryAsync(uint32_t const* frameId) {
    uint32_t fistFrame = *frameId;
    uint32_t endFrame = fistFrame + GetOptimalSwapChainTextureNum();
    m_InstanceMaskToMaskedBlasData.clear(); // stop using masked geometry here

    while (*frameId < endFrame)
        Sleep(1);

    OmmGeometryUpdate(m_OmmComputeContext, false);
}

void Sample::RebuildOmmGeometry() {
    NRI.QueueWaitIdle(m_GraphicsQueue);
    OmmGeometryUpdate(m_OmmGraphicsContext, true);
}

void Sample::ReleaseMaskedGeometry() {
    for (auto& resource : m_MaskedBlasses)
        m_OmmHelper.DestroyMaskedGeometry(resource.blas, resource.ommArray);

    m_InstanceMaskToMaskedBlasData.clear();
    m_MaskedBlasses.clear();
    m_OmmHelper.ReleaseGeometryMemory();
}

void Sample::ReleaseBakingResources() {
    for (AlphaTestedGeometry& geometry : m_OmmAlphaGeometry) {
        geometry.bakeDesc = {};
        geometry.buildDesc = {};
    }

    m_OmmRawAlphaChannelForCpuBaker.resize(0);
    m_OmmRawAlphaChannelForCpuBaker.shrink_to_fit();

    // Destroy buffers
    auto DestroyBuffers = [](NRIInterface& nri, nri::Buffer** buffers, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            if (buffers[i]) {
                nri.DestroyBuffer(buffers[i]);
                buffers[i] = nullptr;
            }
        }
    };
    DestroyBuffers(NRI, m_OmmGpuOutputBuffers, (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);
    DestroyBuffers(NRI, m_OmmGpuReadbackBuffers, (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);
    DestroyBuffers(NRI, m_OmmGpuTransientBuffers, OMM_MAX_TRANSIENT_POOL_BUFFERS);

    for (auto& buffer : m_OmmCpuUploadBuffers)
        NRI.DestroyBuffer(buffer);
    m_OmmCpuUploadBuffers.resize(0);
    m_OmmCpuUploadBuffers.shrink_to_fit();

    // Release memories
    for (auto& memory : m_OmmTmpAllocations)
        NRI.FreeMemory(memory);
    m_OmmTmpAllocations.resize(0);
    m_OmmTmpAllocations.shrink_to_fit();

    for (auto& memory : m_OmmBakerAllocations)
        NRI.FreeMemory(memory);
    m_OmmBakerAllocations.resize(0);
    m_OmmBakerAllocations.shrink_to_fit();

    m_OmmHelper.GpuPostBakeCleanUp();
}

std::vector<ommhelper::OmmBakeGeometryDesc*> GetBakingQueue(std::vector<AlphaTestedGeometry>& geometry) {
    std::vector<ommhelper::OmmBakeGeometryDesc*> result = {};
    result.reserve(geometry.size());

    for (size_t i = 0; i < geometry.size(); ++i)
        result.push_back(&geometry[i].bakeDesc);

    return result;
}

bool IsRebuildAvailable(ommhelper::OmmBakeDesc& updated, ommhelper::OmmBakeDesc& current) {
    bool result = false;
    result |= updated.subdivisionLevel != current.subdivisionLevel;
    result |= updated.mipBias != current.mipBias;
    result |= updated.dynamicSubdivisionScale != current.dynamicSubdivisionScale;
    result |= updated.filter != current.filter;
    result |= updated.format != current.format;

    result |= updated.type != current.type;
    if (current.type == ommhelper::OmmBakerType::GPU) {
        result |= updated.gpuFlags.computeOnlyWorkload != current.gpuFlags.computeOnlyWorkload;
        result |= updated.gpuFlags.enablePostBuildInfo != current.gpuFlags.enablePostBuildInfo;
        result |= updated.gpuFlags.enableTexCoordDeduplication != current.gpuFlags.enableTexCoordDeduplication;
        result |= updated.gpuFlags.force32bitIndices != current.gpuFlags.force32bitIndices;
        result |= updated.gpuFlags.enableSpecialIndices != current.gpuFlags.enableSpecialIndices;
        result |= updated.gpuFlags.allow8bitIndices != current.gpuFlags.allow8bitIndices;
    } else {
        result |= updated.mipCount != current.mipCount;
        result |= updated.cpuFlags.enableInternalThreads != current.cpuFlags.enableInternalThreads;
        result |= updated.cpuFlags.enableSpecialIndices != current.cpuFlags.enableSpecialIndices;
        result |= updated.cpuFlags.enableDuplicateDetection != current.cpuFlags.enableDuplicateDetection;
        result |= updated.cpuFlags.enableNearDuplicateDetection != current.cpuFlags.enableNearDuplicateDetection;
        result |= updated.cpuFlags.force32bitIndices != current.cpuFlags.force32bitIndices;
        result |= updated.cpuFlags.allow8bitIndices != current.cpuFlags.allow8bitIndices;
    }

    result |= ((current.enableCache == false) && updated.enableCache);

    return result;
}

void Sample::AppendOmmImguiSettings() {
    static ommhelper::OmmBakeDesc bakeDesc = m_OmmBakeDesc;

    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
    bool isUnfolded = ImGui::CollapsingHeader("VISIBILITY MASKS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PushID("VISIBILITY MASKS");
    {
        if (isUnfolded) {
            if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12) {
#if DXR_OMM
                ImGui::Text("API: DXR");
                if (ImGui::BeginItemTooltip()) {
                    ImGui::Text("OMMs are built using DXR 1.2 API");
                    ImGui::EndTooltip();
                }
#else
                ImGui::Text("API: NvAPI");
                if (ImGui::BeginItemTooltip()) {
                    ImGui::Text("OMMs are built using NvAPI");
                    ImGui::EndTooltip();
                }
#endif
            }
            ImGui::Checkbox("Enable OMMs", &m_EnableOmm);
            ImGui::SameLine();
            ImGui::Text("[Masked Geometry Num: %llu]", m_MaskedBlasses.size());
            ImVec4 color = m_Settings.highLightAhs ? ImVec4(1.0f, 0.0f, 1.0f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Checkbox("Highlight AHS", &m_Settings.highLightAhs);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Checkbox("AHS Dynamic Mip", &m_Settings.ahsDynamicMipSelection);

            ImGui::Checkbox("Only Alpha Tested", &m_ShowOnlyAlphaTestedGeometry);

            ImGui::Separator();
            ImGui::Text("OMM Baking Settings:");

            static const char* ommBakerTypes[] = {
                "GPU",
                "CPU",
            };
            static int ommBakerTypeSelection = (int)bakeDesc.type;
            ImGui::Combo("BakerType", &ommBakerTypeSelection, ommBakerTypes, helper::GetCountOf(ommBakerTypes));

            int32_t maxSubdivisionLevel = 12;
            float maxSubdivisionScale = 12.0f;
            bool isCpuBaker = ommBakerTypeSelection == 1;
            if (isCpuBaker) // if CPU
            {
                ommhelper::CpuBakerFlags& cpuFlags = bakeDesc.cpuFlags;
                ImGui::Checkbox("SpecialIndices", &cpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("InternalThreads", &cpuFlags.enableInternalThreads);

                ImGui::Checkbox("DuplicateDetection", &cpuFlags.enableDuplicateDetection);
                ImGui::SameLine();
                ImGui::Checkbox("NearDuplicateDetection", &cpuFlags.enableNearDuplicateDetection);
            } else // if GPU
            {
                ommhelper::GpuBakerFlags& gpuFlags = bakeDesc.gpuFlags;
                maxSubdivisionLevel = gpuFlags.computeOnlyWorkload ? 12 : 9; // gpu baker in raster mode is limited to level 9
                ImGui::Checkbox("SpecialIndices", &gpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("Compute", &gpuFlags.computeOnlyWorkload);
                ImGui::SameLine();
                bool prevAsyncValue = m_EnableAsync;
                ImGui::Checkbox("Async", &m_EnableAsync);
                if (prevAsyncValue != m_EnableAsync)
                    gpuFlags.computeOnlyWorkload = m_EnableAsync ? true : gpuFlags.computeOnlyWorkload;
                m_EnableAsync = gpuFlags.computeOnlyWorkload && m_EnableAsync;
                maxSubdivisionScale = gpuFlags.computeOnlyWorkload ? maxSubdivisionScale : 9.0f;
            }

            static int ommFormatSelection = (int)bakeDesc.format;
            static const char* ommFormatNames[] = {
                "OC1_2_STATE",
                "OC1_4_STATE",
            };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("OMM Format", &ommFormatSelection, ommFormatNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int ommFilterSelection = (int)bakeDesc.filter;
            static const char* vmFilterNames[] = {
                "Nearest",
                "Linear",
            };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("Alpha Test Filter", &ommFilterSelection, vmFilterNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int mipBias = bakeDesc.mipBias;
            static int mipCount = bakeDesc.mipCount;
            static int subdivisionLevel = bakeDesc.subdivisionLevel;

            static float subdivisionScale = bakeDesc.dynamicSubdivisionScale;
            static bool enableDynamicSubdivisionScale = true;
            if (enableDynamicSubdivisionScale) {
                ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
                ImGui::SliderFloat("Subdivision Scale", &subdivisionScale, 0.1f, maxSubdivisionScale, "%.1f");
                ImGui::PopItemWidth();
                ImGui::SameLine();
            }

            ImGui::Checkbox(enableDynamicSubdivisionScale ? " " : "Enable Subdivision Scale", &enableDynamicSubdivisionScale);
            bakeDesc.dynamicSubdivisionScale = enableDynamicSubdivisionScale ? subdivisionScale : 0.0f;

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            static char buffer[128];
            sprintf(buffer, "Max Subdivision Level [1 : %d] ", maxSubdivisionLevel);
            ImGui::InputInt(buffer, &subdivisionLevel);
            ImGui::PopItemWidth();
            subdivisionLevel = subdivisionLevel < 1 ? 1 : subdivisionLevel;
            subdivisionLevel = subdivisionLevel > maxSubdivisionLevel ? maxSubdivisionLevel : subdivisionLevel;

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            ImGui::InputInt("Mip Bias (if applicable)", &mipBias);
            ImGui::PopItemWidth();
            mipBias = mipBias < 0 ? 0 : mipBias;
            mipBias = mipBias > 15 ? 15 : mipBias;
            static bool enableCaching = bakeDesc.enableCache;

            if (isCpuBaker) {
                ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
                ImGui::InputInt("Mip Count (if applicable)", &mipCount);
                ImGui::PopItemWidth();
                int maxMipRange = OMM_MAX_MIP_NUM - mipBias;
                mipCount = mipCount < 1 ? 1 : mipCount;
                mipCount = mipCount > maxMipRange ? maxMipRange : mipCount;
            }

            bakeDesc.format = ommhelper::OmmFormats(ommFormatSelection);
            bakeDesc.filter = ommhelper::OmmBakeFilter(ommFilterSelection);
            bakeDesc.subdivisionLevel = subdivisionLevel;
            bakeDesc.mipBias = mipBias;
            bakeDesc.mipCount = mipCount;
            bakeDesc.type = ommhelper::OmmBakerType(ommBakerTypeSelection);
            bakeDesc.enableCache = enableCaching;

            bool isRebuildAvailable = IsRebuildAvailable(bakeDesc, m_OmmBakeDesc);

            static std::future<void> asyncUpdateTask = {};
            bool isAsyncActive = false;
            if (asyncUpdateTask.valid())
                isAsyncActive = asyncUpdateTask.wait_for(std::chrono::microseconds(0)) != std::future_status::ready && asyncUpdateTask.valid();

            const static ImU32 greyColor = ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            const static ImU32 greenColor = ImGui::GetColorU32(ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
            const static ImU32 redColor = ImGui::GetColorU32(ImVec4(0.6f, 0.0f, 0.0f, 1.0f));

            static uint32_t frameId = 0;
            bool forceRebuild = frameId == m_OmmBakeDesc.buildFrameId;
            {
                ImU32 buttonColor = isRebuildAvailable ? greenColor : greyColor;
                buttonColor = isAsyncActive ? redColor : buttonColor;

                ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Button, buttonColor);
                if ((ImGui::Button("Bake OMMs") || forceRebuild) && !isAsyncActive) {
                    m_OmmBakeDesc = bakeDesc;

                    bool launchAsyncTask = (m_EnableAsync && !isCpuBaker) || isCpuBaker;
                    if (launchAsyncTask)
                        asyncUpdateTask = std::async(std::launch::async, &Sample::RebuildOmmGeometryAsync, this, &frameId);
                    else
                        RebuildOmmGeometry();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();
                ImGui::Checkbox("Use OMM Cache", &enableCaching);

                if (isAsyncActive)
                    ImGui::ProgressBar(float(m_OmmUpdateProgress) / float(m_OmmAlphaGeometry.size()));
            }
            ++frameId;
        }
    }
    ImGui::PopID();
}

SAMPLE_MAIN(Sample, 0);