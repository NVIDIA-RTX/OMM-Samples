cmake_minimum_required(VERSION 3.30)

include(FetchContent)
include(CMakeDependentOption)

# Architecture
if((CMAKE_GENERATOR_PLATFORM MATCHES "ARM64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

# GitHub actions?
if("$ENV{GITHUB_ACTIONS}" STREQUAL "")
    set(GITHUB_CI FALSE CACHE BOOL "GitHub Actions runner" FORCE)
else()
    set(GITHUB_CI TRUE CACHE BOOL "GitHub Actions runner" FORCE)
    message(STATUS "Detected environment: GitHub Actions...")
endif()

# Options
option(DXR_OMM "Use DXR 1.2 API" ON)

cmake_dependent_option(USE_MINIMAL_DATA "Use minimal '_Data' (90MB)" ON "GITHUB_CI" OFF)
cmake_dependent_option(RTXCR_INTEGRATION "Use RTXCR for hair and skin rendering, download sample scene" ON "NOT GITHUB_CI" OFF)

set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders" CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")

get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if(IS_MULTI_CONFIG)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
endif()

# Create project
project(OMMSample LANGUAGES C CXX)

# Globals?
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# Download dependencies (Packman)
if(WIN32)
    set(PACKMAN_EXT ".cmd")
else()
    set(PACKMAN_EXT ".sh")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32)
        set(PACKMAN_PLATFORM "windows-x86_64")
    else()
        set(PACKMAN_PLATFORM "linux-x86_64")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(PACKMAN_PLATFORM "linux-aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

message("Packman platform: ${PACKMAN_PLATFORM}")

if(USE_MINIMAL_DATA)
    set(NRI_DATA_VERSION 2.3_minimal)
    message("Using minimal '_Data'")
else()
    set(NRI_DATA_VERSION 2.3)
endif()

execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/packman${PACKMAN_EXT}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -p ${PACKMAN_PLATFORM} -t nri_data_version=${NRI_DATA_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PACKMAN_RESULT
)

if(NOT "${PACKMAN_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Packman failed(code = ${PACKMAN_RESULT})")
endif()

# Download dependencies
set(DEPS)

if(NOT TARGET ShaderMake)
    # ShaderMake
    option(SHADERMAKE_TOOL "" OFF)

    FetchContent_Declare(
        shadermake
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/ShaderMake.git
        GIT_TAG main
        GIT_SHALLOW 1
    )
    list(APPEND DEPS shadermake)
endif()

# SHARC
FetchContent_Declare(
    sharc
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/SHARC.git
    GIT_TAG v1.6.0.0
    GIT_SHALLOW 1
)
list(APPEND DEPS sharc)

# RTXCR
if(RTXCR_INTEGRATION)
    FetchContent_Declare(
        rtxcr
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/RTXCR-Material-Library.git
        GIT_TAG v1.1.0
        GIT_SHALLOW 1
    )
    list(APPEND DEPS rtxcr)

    FetchContent_Declare(
        claire
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/RTXCR-Assets.git
        GIT_TAG v1.1.0
        GIT_SHALLOW 1
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_Data/Scenes/Claire"
    )
    list(APPEND DEPS claire)
endif()

if(DEPS)
    message("Downloading dependencies:")
    message(STATUS "${DEPS} ...")

    FetchContent_MakeAvailable(${DEPS})
endif()

# Compile options
if(NOT IS_ARM64)
    set(SIMD -mssse3)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS ${SIMD} -Wextra -Wno-missing-field-initializers)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS ${SIMD} -Wextra -Wno-missing-field-initializers)
elseif(MSVC)
    set(COMPILE_OPTIONS /W4 /WX
        /wd4324 # padding was added at the end of a structure because you specified an alignment specifier
        /wd4834 # discarding return value of function with 'nodiscard' attribute
    )
else()
    message(WARNING "Unknown compiler!")
endif()

# Compile definitions
if(WIN32)
    set(COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# External/NRIFramework
set(NRI_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
option(NRI_ENABLE_NONE_SUPPORT "" OFF)
option(NRI_ENABLE_D3D11_SUPPORT "" OFF)
option(NRI_ENABLE_NIS_SDK "" ON)
if(NOT IS_ARM64)
option(NRI_ENABLE_NGX_SDK "" ON)
option(NRI_ENABLE_FFX_SDK "" ON)
option(NRI_ENABLE_XESS_SDK "" ON)
endif()

add_subdirectory("External/NRIFramework")

# External/NRD
set(NRD_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(NRD_NORMAL_ENCODING "2" CACHE STRING "")
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "")
set(NRD_EMBEDS_DXBC_SHADERS "0" CACHE STRING "")
option(NRD_SUPPORTS_CHECKERBOARD "" OFF)
option(NRD_SUPPORTS_HISTORY_CONFIDENCE "" OFF)
option(NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX "" OFF)
option(NRD_SUPPORTS_BASECOLOR_METALNESS "" OFF)

add_subdirectory("External/NRD")

# Opacity Micro-Maps
set(OMM_VK_S_SHIFT 0 CACHE STRING "OMM_VK_S_SHIFT")
set(OMM_VK_T_SHIFT 128 CACHE STRING "OMM_VK_T_SHIFT")
set(OMM_VK_B_SHIFT 32 CACHE STRING "OMM_VK_B_SHIFT")
set(OMM_VK_U_SHIFT 64 CACHE STRING "OMM_VK_U_SHIFT")
set(OMM_PROJECT_FOLDER "Opacity Micro-Map SDK")
set(OMM_ENABLE_TESTS OFF CACHE BOOL "")
set(OMM_BUILD_VIEWER OFF CACHE BOOL "")
set(OMM_INTEGRATION_LAYER_NVRHI OFF CACHE BOOL "")
set(OMM_STATIC_LIBRARY OFF CACHE BOOL "")
set(OMM_OUTPUT_BIN_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_subdirectory("External/Opacity-MicroMap-SDK")

# Move dependencies into "External"
function(fix_folders _DIRECTORY _FOLDER)
    get_property(_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)

    foreach(_TARGET IN LISTS _TARGETS)
        get_property(_FOLDER_ORIGINAL TARGET ${_TARGET} PROPERTY FOLDER)
        set_target_properties(${_TARGET} PROPERTIES FOLDER "${_FOLDER}/${_FOLDER_ORIGINAL}")
    endforeach()

    get_property(SUBDIRS DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)

    foreach(SUBDIR IN LISTS SUBDIRS)
        fix_folders("${SUBDIR}" "${_FOLDER}")
    endforeach()
endfunction()

fix_folders("External/NRIFramework" "External")
fix_folders("External/NRD" "External")

# Get source directories for 3rd parties
get_target_property(ML_SOURCE_DIR MathLib SOURCE_DIR)
get_target_property(NRI_SOURCE_DIR NRI SOURCE_DIR)
get_target_property(NRD_SOURCE_DIR NRD SOURCE_DIR)

# OMM sample
file(GLOB OMM_SAMPLE_SOURCE "Source/*.cpp" "Source/*.hpp")
source_group("" FILES ${OMM_SAMPLE_SOURCE})

add_executable(${PROJECT_NAME} ${OMM_SAMPLE_SOURCE})

if (DXR_OMM)
# DXR 1.2 API. AgilitySDK is handled by NRI
    target_compile_definitions(${PROJECT_NAME} PRIVATE DXR_OMM=1)

    set(ASDK_DEP)

    FetchContent_Declare(
        agilitysdk
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        URL https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.${NRI_AGILITY_SDK_VERSION_MAJOR}.${NRI_AGILITY_SDK_VERSION_MINOR}
    )
    list(APPEND ASDK_DEP agilitysdk)

    if(ASDK_DEP)
        message(STATUS "Downloading ${ASDK_DEP}")
        FetchContent_MakeAvailable(${ASDK_DEP})
        target_include_directories(${PROJECT_NAME} PRIVATE "${agilitysdk_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "AgilitySDK Failed")
    endif()
else()
    # NVAPI
    set(NVAPI_DEP)
    FetchContent_Declare(
        nvapi
        GIT_REPOSITORY https://github.com/NVIDIA/nvapi.git
        GIT_TAG main
        GIT_SHALLOW 1
    )
    list(APPEND NVAPI_DEP nvapi)

    if(NVAPI_DEP)
        message(STATUS "Downloading ${NVAPI_DEP}")
        FetchContent_MakeAvailable(${NVAPI_DEP})
    endif()

    find_library(NVAPI_LIB NAMES nvapi64 nvapi PATHS "${nvapi_SOURCE_DIR}/amd64" REQUIRED)
    if (NVAPI_LIB)
        set(NVAPI_LIB_PATH "${nvapi_SOURCE_DIR}/amd64/nvapi64.lib")
        target_link_libraries(${PROJECT_NAME} PRIVATE ${NVAPI_LIB_PATH})
        target_include_directories(${PROJECT_NAME} PRIVATE "${nvapi_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "Can't find NVAPI64 LIB")
    endif()
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    "Source"
    "External"
    "External/omm_bake/libraries/omm-lib/include"
    "$ENV{VULKAN_SDK}/include"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${COMPILE_DEFINITIONS}
    PROJECT_NAME=${PROJECT_NAME}
    SHADERMAKE_PATH=${SHADERMAKE_PATH}
    SHADERMAKE_DXC_PATH=${SHADERMAKE_DXC_PATH}
    SHADERMAKE_DXC_VK_PATH=${SHADERMAKE_DXC_VK_PATH}
    ML_SOURCE_DIR=${ML_SOURCE_DIR}
    NRI_SOURCE_DIR=${NRI_SOURCE_DIR}/Include
    NRD_SOURCE_DIR=${NRD_SOURCE_DIR}/Shaders/Include
    SHARC_SOURCE_DIR=${sharc_SOURCE_DIR}/include
    RTXCR_SOURCE_DIR=${rtxcr_SOURCE_DIR}/shaders/include/rtxcr
    RTXCR_INTEGRATION=$<BOOL:${RTXCR_INTEGRATION}>
)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    NRIFramework
    NRD
    NRDIntegration
    omm-lib
)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS} pthread X11)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "Sample"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Copy arguments for Visual Studio Smart Command Line Arguments extension
if(WIN32 AND MSVC)
    configure_file(.args "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.args.json" COPYONLY)
endif()

# Shaders
file(GLOB_RECURSE SHADERS
    "Shaders/*.hlsl"
    "Shaders/*.hlsli"
)
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

set(SHADERMAKE_GENERAL_ARGS
    --project "OMMSample"
    --compactProgress
    --flatten
    --stripReflection
    --WX
    --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128 # TODO: must be in sync with "VK_BINDING_OFFSETS"
    --binary
    --shaderModel 6_6
    --sourceDir "Shaders"
    --ignoreConfigDir
    -c "Shaders/Shaders.cfg"
    -o "${SHADER_OUTPUT_PATH}"
    -I "Shaders"
    -I "External"
    -I "${ML_SOURCE_DIR}"
    -I "${NRD_SOURCE_DIR}/Shaders/Include"
    -I "${NRI_SOURCE_DIR}/Include"
    -I "${sharc_SOURCE_DIR}/include"
    -I "${rtxcr_SOURCE_DIR}/shaders/include/rtxcr"
    -D RTXCR_INTEGRATION=$<BOOL:${RTXCR_INTEGRATION}>
    --relaxedInclude "HairFarFieldBCSDF.hlsli"
    --relaxedInclude "SubsurfaceScattering.hlsli"
)

set(SHADERMAKE_COMMANDS)
if(NRI_ENABLE_D3D12_SUPPORT)
    list(APPEND SHADERMAKE_COMMANDS COMMAND ${SHADERMAKE_PATH} -p DXIL --compiler "${SHADERMAKE_DXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
endif()
if(NRI_ENABLE_VK_SUPPORT)
    list(APPEND SHADERMAKE_COMMANDS COMMAND ${SHADERMAKE_PATH} -p SPIRV --compiler "${SHADERMAKE_DXC_VK_PATH}" ${SHADERMAKE_GENERAL_ARGS})
endif()
list(APPEND SHADERMAKE_COMMANDS "") # fix for "The system cannot find the batch label specified - VCEnd"

add_custom_target(${PROJECT_NAME}Shaders ALL
    ${SHADERMAKE_COMMANDS}
    DEPENDS ShaderMake
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
    SOURCES ${SHADERS}
)

# Opacity Micro-Map integration files

if (TARGET omm-lib)
    file(GLOB VM_INTEGRATION_FILES "Source/VisibilityMasks/*.h" "Source/VisibilityMasks/*.cpp")
    source_group("Omm Helper" FILES ${VM_INTEGRATION_FILES})
    target_sources(${PROJECT_NAME}  PRIVATE ${VM_INTEGRATION_FILES})

    file(GLOB PROFILER_FILES "Source/Profiler/*.hpp")
    source_group("Profiler" FILES ${PROFILER_FILES})
    target_sources(${PROJECT_NAME}  PRIVATE ${PROFILER_FILES})
endif()

set_target_properties(${PROJECT_NAME}Shaders PROPERTIES FOLDER "Sample")
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}Shaders)
