/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "MathLib/STL.hlsli"
#include "NRD/Shaders/Include/NRD.hlsli"

#include "BindingBridge.hlsli"

//===============================================================
// GLOSSARY
//===============================================================
/*
Names:
- V - view vector
- N - normal
- X - point position

Modifiers:
- v - view space
- 0..N - hit index ( 0 - primary ray )
*/

//=============================================================================================
// SETTINGS
//=============================================================================================

#define NRD_MODE                            NORMAL // NORMAL, OCCLUSION, SH, DIRECTIONAL_OCCLUSION

#define USE_SIMPLEX_LIGHTING_MODEL          0
#define USE_IMPORTANCE_SAMPLING             1
#define USE_SANITIZATION                    0 // NRD sample is NAN/INF free
#define USE_PSR                             1
#define USE_SIMULATED_MATERIAL_ID_TEST      0 // for "material ID" support debugging
#define USE_SIMULATED_FIREFLY_TEST          0 // "anti-firefly" debugging

#define BRDF_ENERGY_THRESHOLD               0.001
#define AMBIENT_FADE                        ( -0.001 * gUnitToMetersMultiplier * gUnitToMetersMultiplier )
#define TAA_HISTORY_SHARPNESS               0.5 // [0; 1], 0.5 matches Catmull-Rom
#define TAA_MAX_HISTORY_WEIGHT              0.95
#define TAA_MIN_HISTORY_WEIGHT              0.1
#define TAA_MOTION_MAX_REUSE                0.1
#define MAX_MIP_LEVEL                       11.0
#define IMPORTANCE_SAMPLE_NUM               16

//=============================================================================================
// CONSTANTS
//=============================================================================================

// NRD variant
#define NORMAL                              0
#define OCCLUSION                           1
#define SH                                  2
#define DIRECTIONAL_OCCLUSION               3

// Denoiser
#define REBLUR                              0
#define RELAX                               1

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
#define SHOW_WORLD_UNITS                    10
#define SHOW_INSTANCE_INDEX                 11
#define SHOW_MIP_PRIMARY                    12
#define SHOW_MIP_SPECULAR                   13

// Predefined material override
#define MAT_GYPSUM                          1
#define MAT_COBALT                          2

// Other
#define FP16_MAX                            65504.0
#define INF                                 1e5

//===============================================================
// FP16
//===============================================================

#ifdef NRD_COMPILER_DXC
    #define half_float float16_t
    #define half_float2 float16_t2
    #define half_float3 float16_t3
    #define half_float4 float16_t4
#else
    #define half_float float
    #define half_float2 float2
    #define half_float3 float3
    #define half_float4 float4
#endif

//===============================================================
// RESOURCES
//===============================================================

NRI_RESOURCE( cbuffer, globalConstants, b, 0, 0 )
{
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float3 gSunDirection;
    float gExposure;
    float3 gCameraOrigin;
    float gMipBias;
    float3 gTrimmingParams;
    float gEmissionIntensity;
    float3 gViewDirection;
    float gOrthoMode;
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
    float gNearZ;
    float gAmbientAccumSpeed;
    float gAmbient;
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
    float gReference;
    float gUsePrevFrame;
    float gMinProbability;
    uint gDenoiserType;
    uint gDisableShadowsAndEnableImportanceSampling; // TODO: remove - modify GetSunIntensity to return 0 if sun is below horizon
    uint gOnScreen;
    uint gFrameIndex;
    uint gForcedMaterial;
    uint gUseNormalMap;
    uint gIsWorldSpaceMotionEnabled;
    uint gTracingMode;
    uint gSampleNum;
    uint gBounceNum;
    uint gTAA;
    uint gResolve;
    uint gPSR;
    uint gValidation;
    uint gHighlightAhs;
    uint gAhsDynamicMip;

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
    uint gNisInputViewportOriginX;
    uint gNisInputViewportOriginY;
    uint gNisInputViewportWidth;
    uint gNisInputViewportHeight;
    uint gNisOutputViewportOriginX;
    uint gNisOutputViewportOriginY;
    uint gNisOutputViewportWidth;
    uint gNisOutputViewportHeight;
};

NRI_RESOURCE( SamplerState, gLinearMipmapLinearSampler, s, 0, 0 );
NRI_RESOURCE( SamplerState, gLinearMipmapNearestSampler, s, 1, 0 );
NRI_RESOURCE( SamplerState, gLinearSampler, s, 2, 0 );
NRI_RESOURCE( SamplerState, gNearestSampler, s, 3, 0 );

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

        // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues due to FP16 rounding negative effects
        motion.z = viewZprev - viewZ;
    }

    return motion;
}

// IMPORTANT: requires STL::Rng::Hash::Initialize
float3 ApplyExposure( float3 Lsum, bool convertToLDR = true )
{
    // Exposure
    if( gOnScreen <= SHOW_DENOISED_SPECULAR )
    {
        Lsum *= gExposure;

        // Dithering
        float rnd = STL::Rng::Hash::GetFloat( );
        float luma = STL::Color::Luminance( Lsum );
        float amplitude = lerp( 0.4, 1.0 / 1024.0, STL::Math::Sqrt01( luma ) );
        float dither = 1.0 + ( rnd - 0.5 ) * amplitude;
        Lsum *= dither;
    }

    // Tonemap
    if( convertToLDR && gOnScreen == SHOW_FINAL )
        Lsum = STL::Color::HdrToLinear_Uncharted( Lsum );

    // Conversion
    if( convertToLDR && ( gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR ) )
        Lsum = STL::Color::LinearToSrgb( Lsum );

    return Lsum;
}

float3 BicubicFilterNoCorners( Texture2D<float3> tex, SamplerState samp, float2 samplePos, float2 invTextureSize, compiletime const float sharpness )
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

    return STL::Color::GammaToLinear( saturate( skyColor ) ) * SKY_INTENSITY + GetSunIntensity( v, sunDirection, tanAngularRadius );
}
