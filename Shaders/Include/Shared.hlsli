/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

//=============================================================================================
// SETTINGS
//=============================================================================================

// Fused or separate denoising selection
//      0 - DIFFUSE and SPECULAR
//      1 - DIFFUSE_SPECULAR
#define NRD_COMBINED                        1

//      NORMAL                - common (non specialized) denoisers
//      OCCLUSION             - OCCLUSION (ambient or specular occlusion only) denoisers
//      SH                    - SH (spherical harmonics or spherical gaussian) denoisers
//      DIRECTIONAL_OCCLUSION - DIRECTIONAL_OCCLUSION (ambient occlusion in SH mode) denoisers
#define NRD_MODE                            NORMAL // NORMAL, OCCLUSION, SH, DIRECTIONAL_OCCLUSION

// Default = 1
#define USE_IMPORTANCE_SAMPLING             1
#define USE_PSR                             1 // allow primary surface replacement

// Default = 0
#define USE_SIMPLEX_LIGHTING_MODEL          0
#define USE_SANITIZATION                    0 // NRD sample is NAN/INF free
#define USE_SIMULATED_MATERIAL_ID_TEST      0 // for "material ID" support debugging
#define USE_SIMULATED_FIREFLY_TEST          0 // "anti-firefly" debugging
#define USE_RUSSIAN_ROULETTE                0 // bad practice for real-time denoising
#define USE_DRS_STRESS_TEST                 0 // test for verifying that NRD doesn't touch data outside of DRS rectangle
#define USE_INF_STRESS_TEST                 0 // test for verifying that NRD doesn't touch data outside of denoising range
#define USE_ANOTHER_COBALT                  0 // another cobalt variant
#define USE_PUDDLES                         0 // add puddles
#define USE_RANDOMIZED_ROUGHNESS            0 // randomize roughness ( a common case in games )
#define USE_LOAD                            0 // Load vs SampleLevel

#define THROUGHPUT_THRESHOLD                0.001
#define PSR_THROUGHPUT_THRESHOLD            0.0 // TODO: even small throughput can produce a bright spot if incoming radiance is huge
#define MAX_MIP_LEVEL                       11.0
#define IMPORTANCE_SAMPLES_NUM              16
#define TAA_HISTORY_SHARPNESS               0.5 // [0; 1], 0.5 matches Catmull-Rom
#define TAA_MAX_HISTORY_WEIGHT              0.95
#define TAA_MIN_HISTORY_WEIGHT              0.1
#define TAA_MOTION_MAX_REUSE                0.1
#define SPEC_LOBE_ENERGY                    0.95 // trimmed to 95%
#define AMBIENT_FADE                        ( -0.001 * gUnitToMetersMultiplier * gUnitToMetersMultiplier )

//=============================================================================================
// CONSTANTS
//=============================================================================================

// NRD variant
#define NORMAL                              0
#define OCCLUSION                           1
#define SH                                  2
#define DIRECTIONAL_OCCLUSION               3

// Denoiser
#define DENOISER_REBLUR                     0
#define DENOISER_RELAX                      1
#define DENOISER_REFERENCE                  2

// Resolution
#define RESOLUTION_FULL                     0
#define RESOLUTION_FULL_PROBABILISTIC       1
#define RESOLUTION_HALF                     2

// What is on screen?
#define SHOW_FINAL                          0
#define SHOW_DENOISED_DIFFUSE               1
#define SHOW_DENOISED_SPECULAR              2
#define SHOW_AMBIENT_OCCLUSION              3
#define SHOW_SPECULAR_OCCLUSION             4
#define SHOW_SHADOW                         5
#define SHOW_BASE_COLOR                     6
#define SHOW_NORMAL                         7
#define SHOW_ROUGHNESS                      8
#define SHOW_METALNESS                      9
#define SHOW_MATERIAL_ID                    10
#define SHOW_PSR_THROUGHPUT                 11
#define SHOW_WORLD_UNITS                    12
#define SHOW_INSTANCE_INDEX                 13
#define SHOW_UV                             14
#define SHOW_CURVATURE                      15
#define SHOW_MIP_PRIMARY                    16
#define SHOW_MIP_SPECULAR                   17

// Predefined material override
#define MATERIAL_GYPSUM                     1
#define MATERIAL_COBALT                     2

// Material ID
#define MATERIAL_ID_DEFAULT                 0
#define MATERIAL_ID_METAL                   1
#define MATERIAL_ID_PSR                     2
#define MATERIAL_ID_HAIR                    3

// Mip mode
#define MIP_VISIBILITY                      0 // for visibility: emission, shadow and alpha mask
#define MIP_LESS_SHARP                      1 // for normal
#define MIP_SHARP                           2 // for albedo and roughness

// Other
#define FP16_MAX                            65504.0
#define INF                                 1e5

#define MORPH_MAX_ACTIVE_TARGETS_NUM        8u
#define MORPH_ELEMENTS_PER_ROW_NUM          4
#define MORPH_ROWS_NUM                      ( MORPH_MAX_ACTIVE_TARGETS_NUM / MORPH_ELEMENTS_PER_ROW_NUM )

// Instance flags
#define FLAG_FIRST_BIT                      26 // this + number of flags must be <= 32
#define NON_FLAG_MASK                       ( ( 1 << FLAG_FIRST_BIT ) - 1 )

#define FLAG_DEFAULT                        0x01 // always set
#define FLAG_TRANSPARENT                    0x02 // transparent
#define FLAG_FORCED_EMISSION                0x04 // animated emissive cube
#define FLAG_STATIC                         0x08 // no velocity
#define FLAG_DEFORMABLE                     0x10 // local animation
#define FLAG_HAIR                           0x20 // hair

#define GEOMETRY_ALL                        0xFF
#define GEOMETRY_ONLY_TRANSPARENT           ( FLAG_TRANSPARENT )
#define GEOMETRY_IGNORE_TRANSPARENT         ( ~FLAG_TRANSPARENT )

//===============================================================
// STRUCTS
//===============================================================

#if( defined( __cplusplus ) )
    // IMPORTANT: sizeof( float3 ) == 16 in C++ code!
    #define float16_t2 uint32_t
    #define float16_t4 uint2
#endif

// IMPORTANT: must match utils::MorphTargetVertex
struct MorphVertex
{
    float16_t4 position;
    float16_t2 N;
    float16_t2 T;
};

struct MorphedAttributes
{
    float16_t2 N;
    float16_t2 T;
};

struct MorphedPrimitivePrevData
{
    float4 position0;
    float4 position1;
    float4 position2;
};

struct PrimitiveData
{
    float16_t2 uv0;
    float16_t2 uv1;
    float16_t2 uv2;
    float16_t2 n0;

    float16_t2 n1;
    float16_t2 n2;
    float16_t2 t0;
    float16_t2 t1;

    float16_t2 t2;
    float16_t2 curvature0_curvature1;
    float16_t2 curvature2_bitangentSign;
    float worldToUvUnits;
};

struct InstanceData
{
    // For static: mObjectToWorld
    // For rigid dynamic: mWorldToWorldPrev
    // For deformable dynamic: mObjectToWorldPrev
    float4 mOverloadedMatrix0;
    float4 mOverloadedMatrix1;
    float4 mOverloadedMatrix2;

    float4 baseColorAndMetalnessScale;
    float4 emissionAndRoughnessScale;

    uint32_t textureOffsetAndFlags;
    uint32_t primitiveOffset;
    uint32_t morphedPrimitiveOffset;

    // TODO: handling object scale embedded into the transformation matrix (assuming uniform scale)
    // TODO: sign represents triangle winding
    float invScale;
};

//===============================================================
// RESOURCES
//===============================================================

#include "BindingBridge.hlsli"

NRI_RESOURCE( cbuffer, GlobalConstants, b, 0, 0 )
{
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float4 gSunDirection_gExposure;
    float4 gCameraOrigin_gMipBias;
    float4 gViewDirection_gOrthoMode;
    float4 gHairBaseColorOverride; // w is alpha or blend factor
    float2 gHairBetasOverride;
    float2 gWindowSize;
    float2 gInvWindowSize;
    float2 gOutputSize;
    float2 gInvOutputSize;
    float2 gRenderSize;
    float2 gInvRenderSize;
    float2 gRectSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gEmissionIntensity;
    float gNearZ;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gTransparent;
    float gPrevFrameConfidence;
    float gMinProbability;
    float gUnproject;
    float gAperture;
    float gFocalDistance;
    float gFocalLength;
    uint32_t gDenoiserType;
    uint32_t gDisableShadowsAndEnableImportanceSampling; // TODO: remove - modify GetSunIntensity to return 0 if sun is below horizon
    uint32_t gOnScreen;
    uint32_t gFrameIndex;
    uint32_t gForcedMaterial;
    uint32_t gUseNormalMap;
    uint32_t gIsWorldSpaceMotionEnabled;
    uint32_t gTracingMode;
    uint32_t gSampleNum;
    uint32_t gBounceNum;
    uint32_t gTAA;
    uint32_t gResolve;
    uint32_t gPSR;
    uint32_t gValidation;
    uint32_t gTrimLobe;
    uint32_t gHighlightAhs;
    uint32_t gAhsDynamicMip;

    // Ambient
    float gAmbientMaxAccumulatedFramesNum;
    float gAmbient;

    // NIS
    float gNisDetectRatio;
    float gNisDetectThres;
    float gNisMinContrastRatio;
    float gNisRatioNorm;
    float gNisContrastBoost;
    float gNisEps;
    float gNisSharpStartY;
    float gNisSharpScaleY;
    float gNisSharpStrengthMin;
    float gNisSharpStrengthScale;
    float gNisSharpLimitMin;
    float gNisSharpLimitScale;
    float gNisScaleX;
    float gNisScaleY;
    float gNisDstNormX;
    float gNisDstNormY;
    float gNisSrcNormX;
    float gNisSrcNormY;
    uint32_t gNisInputViewportOriginX;
    uint32_t gNisInputViewportOriginY;
    uint32_t gNisInputViewportWidth;
    uint32_t gNisInputViewportHeight;
    uint32_t gNisOutputViewportOriginX;
    uint32_t gNisOutputViewportOriginY;
    uint32_t gNisOutputViewportWidth;
    uint32_t gNisOutputViewportHeight;
};

NRI_RESOURCE( cbuffer, MorphMeshUpdateVerticesConstants, b, 0, 3 )
{
    uint4 gIndices[ MORPH_ROWS_NUM ];
    float4 gWeights[ MORPH_ROWS_NUM ];

    uint32_t gNumWeights;
    uint32_t gNumVertices;
    uint32_t gPositionCurrFrameOffset;
    uint32_t gAttributesOutputOffset;
};

NRI_RESOURCE( cbuffer, MorphMeshUpdatePrimitivesConstants, b, 0, 3 )
{
    uint2 gPositionFrameOffsets;
    uint32_t gNumPrimitives;
    uint32_t gIndexOffset;

    uint32_t gAttributesOffset;
    uint32_t gPrimitiveOffset;
    uint32_t gMorphedPrimitiveOffset;
};

#if( !defined( __cplusplus ) )

#include "MathLib/STL.hlsli"
#include "NRD/Shaders/Include/NRD.hlsli"

NRI_RESOURCE( SamplerState, gLinearMipmapLinearSampler, s, 0, 0 );
NRI_RESOURCE( SamplerState, gLinearMipmapNearestSampler, s, 1, 0 );
NRI_RESOURCE( SamplerState, gNearestMipmapNearestSampler, s, 2, 0 );

#define gLinearSampler gLinearMipmapLinearSampler
#define gNearestSampler gNearestMipmapNearestSampler

//=============================================================================================
// MISC
//=============================================================================================

// Taken out from NRD
float GetSpecMagicCurve( float roughness )
{
    float f = 1.0 - exp2( -200.0 * roughness * roughness );
    f *= STL::Math::Pow01( roughness, 0.5 );

    return f;
}

// Returns 3D motion in world space or 2.5D motion in screen space
float3 GetMotion( float3 X, float3 Xprev )
{
    float3 motion = Xprev - X;

    if( !gIsWorldSpaceMotionEnabled )
    {
        float viewZ = STL::Geometry::AffineTransform( gWorldToView, X ).z;
        float2 sampleUv = STL::Geometry::GetScreenUv( gWorldToClip, X );

        float viewZprev = STL::Geometry::AffineTransform( gWorldToViewPrev, Xprev ).z;
        float2 sampleUvPrev = STL::Geometry::GetScreenUv( gWorldToClipPrev, Xprev );

        // IMPORTANT: scaling to "pixel" unit significantly improves utilization of FP16
        motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;

        // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues caused by FP16 rounding negative effects
        motion.z = viewZprev - viewZ;
    }

    return motion;
}

// IMPORTANT: requires STL::Rng::Hash::Initialize
float3 ApplyExposure( float3 Lsum, bool applyToneMap )
{
    // Exposure
    if( gOnScreen <= SHOW_DENOISED_SPECULAR )
    {
        Lsum *= gSunDirection_gExposure.w;

        // Dithering
        float rnd = STL::Rng::Hash::GetFloat( );
        float luma = STL::Color::Luminance( Lsum );
        float amplitude = lerp( 0.4, 1.0 / 1024.0, STL::Math::Sqrt01( luma ) );
        float dither = 1.0 + ( rnd - 0.5 ) * amplitude;
        Lsum *= dither;
    }

    // Tonemap
    if( applyToneMap && gOnScreen == SHOW_FINAL )
        Lsum = STL::Color::HdrToLinear_Uncharted( Lsum );

    return Lsum;
}

float3 BicubicFilterNoCorners( Texture2D<float3> tex, SamplerState samp, float2 samplePos, float2 invTextureSize, float sharpness )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5;
    float2 f = samplePos - centerPos;
    float2 f2 = f * f;
    float2 f3 = f * f2;
    float2 w0 = -sharpness * f3 + 2.0 * sharpness * f2 - sharpness * f;
    float2 w1 = ( 2.0 - sharpness ) * f3 - ( 3.0 - sharpness ) * f2 + 1.0;
    float2 w2 = -( 2.0 - sharpness ) * f3 + ( 3.0 - 2.0 * sharpness ) * f2 + sharpness * f;
    float2 w3 = sharpness * f3 - sharpness * f2;
    float2 wl2 = w1 + w2;
    float2 tc2 = invTextureSize * ( centerPos + w2 * STL::Math::PositiveRcp( wl2 ) );
    float2 tc0 = invTextureSize * ( centerPos - 1.0 );
    float2 tc3 = invTextureSize * ( centerPos + 2.0 );

    float w = wl2.x * w0.y;
    float3 color = tex.SampleLevel( samp, float2( tc2.x, tc0.y ), 0 ) * w;
    float sum = w;

    w = w0.x  * wl2.y;
    color += tex.SampleLevel( samp, float2( tc0.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * wl2.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc2.y ), 0 ) * w;
    sum += w;

    w = w3.x  * wl2.y;
    color += tex.SampleLevel( samp, float2( tc3.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * w3.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc3.y ), 0 ) * w;
    sum += w;

    color *= STL::Math::PositiveRcp( sum );

    return color;
}

void GetCameraRay( out float3 origin, out float3 direction, float2 sampleUv )
{
    // https://www.slideshare.net/TiagoAlexSousa/graphics-gems-from-cryengine-3-siggraph-2013 ( slides 23+ )

    // Pinhole ray
    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gViewDirection_gOrthoMode.w );
    direction = normalize( Xv );

    // Distorted ray
    float2 rnd = STL::Rng::Hash::GetFloat2( );
    rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
    Xv.xy += rnd * gAperture;

    float3 Fv = direction * gFocalDistance; // z-plane
    #if 0
        Fv /= dot( vForward, direction ); // radius
    #endif

    origin = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    direction = gViewDirection_gOrthoMode.w == 0.0 ? normalize( STL::Geometry::RotateVector( gViewToWorld, Fv - Xv ) ) : -gViewDirection_gOrthoMode.xyz;
}

float GetCircleOfConfusion( float distance ) // diameter
{
    float F = gFocalLength; // focal lenght ( deducted from FOV )
    float A = gAperture; // aperture diameter
    float P = gFocalDistance; // focal distance

    return gViewDirection_gOrthoMode.w == 0.0 ? abs( A * ( F * ( P - distance ) ) / ( distance * ( P - F ) ) ) : A;
}

//=============================================================================================
// VERY SIMPLE SKY MODEL
//=============================================================================================

#define SKY_INTENSITY 1.0
#define SUN_INTENSITY 8.0

float3 GetSunIntensity( float3 v, float3 sunDirection, float tanAngularRadius )
{
    float b = dot( v, sunDirection );
    float d = length( v - sunDirection * b );

    float glow = saturate( 1.015 - d );
    glow *= b * 0.5 + 0.5;
    glow *= 0.6;

    float a = STL::Math::Sqrt01( 1.0 - b * b ) / b;
    float sun = 1.0 - STL::Math::SmoothStep( tanAngularRadius * 0.9, tanAngularRadius * 1.66, a );
    sun *= float( b > 0.0 );
    sun *= 1.0 - STL::Math::Pow01( 1.0 - v.z, 4.85 );
    sun *= STL::Math::SmoothStep( 0.0, 0.1, sunDirection.z );
    sun += glow;

    float3 sunColor = lerp( float3( 1.0, 0.6, 0.3 ), float3( 1.0, 0.9, 0.7 ), STL::Math::Sqrt01( sunDirection.z ) );
    sunColor *= saturate( sun );

    sunColor *= STL::Math::SmoothStep( -0.01, 0.05, sunDirection.z );

    return STL::Color::GammaToLinear( sunColor ) * SUN_INTENSITY;
}

float3 GetSkyIntensity( float3 v, float3 sunDirection, float tanAngularRadius )
{
    float atmosphere = sqrt( 1.0 - saturate( v.z ) );

    float scatter = pow( saturate( sunDirection.z ), 1.0 / 15.0 );
    scatter = 1.0 - clamp( scatter, 0.8, 1.0 );

    float3 scatterColor = lerp( float3( 1.0, 1.0, 1.0 ), float3( 1.0, 0.3, 0.0 ) * 1.5, scatter );
    float3 skyColor = lerp( float3( 0.2, 0.4, 0.8 ), float3( scatterColor ), atmosphere / 1.3 );
    skyColor *= saturate( 1.0 + sunDirection.z );

    float ground = 0.5 + 0.5 * STL::Math::SmoothStep( -1.0, 0.0, v.z );
    skyColor *= ground;

    return STL::Color::GammaToLinear( saturate( skyColor ) ) * SKY_INTENSITY + GetSunIntensity( v, sunDirection, tanAngularRadius );
}

#endif
