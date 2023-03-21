/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmSample.hpp"

float CalculateWorldToUvUnits(const utils::Scene& scene, uint32_t vertexOffset, uint32_t indexOffset)
{
    const utils::UnpackedVertex& v0 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset]];
    const utils::UnpackedVertex& v1 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + 1]];
    const utils::UnpackedVertex& v2 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + 2]];

    float3 p0(v0.position);
    float3 p1(v1.position);
    float3 p2(v2.position);

    float3 edge20 = p2 - p0;
    float3 edge10 = p1 - p0;
    float3 triangleNormal = Cross(edge20, edge10);
    float worldArea = Max(Length(triangleNormal), 1e-9f);

    float3 uvEdge20 = float3(v2.uv[0], v2.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
    float3 uvEdge10 = float3(v1.uv[0], v1.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
    float uvArea = Length(Cross(uvEdge20, uvEdge10));

    return uvArea == 0 ? 1.0f : Sqrt(uvArea / worldArea);
}

void Sample::PushVertex(utils::Scene& scene, float positionX, float positionY, float positionZ, float texCoordU, float texCoordV)
{
    utils::UnpackedVertex unpackedVertex = {};
    unpackedVertex.position[0] = positionX;
    unpackedVertex.position[1] = positionY;
    unpackedVertex.position[2] = positionZ;
    unpackedVertex.uv[0] = texCoordU;
    unpackedVertex.uv[1] = texCoordV;

    utils::Vertex vertex = {};
    for (size_t i = 0; i < helper::GetCountOf(vertex.position); i++)
        vertex.position[i] = unpackedVertex.position[i];
    vertex.uv = Packed::sf2_to_h2(unpackedVertex.uv[0], unpackedVertex.uv[1]);

    scene.unpackedVertices.push_back(unpackedVertex);
    scene.vertices.push_back(vertex);
}

void Sample::ComputePrimitiveNormal(utils::Scene& scene, uint32_t vertexOffset, uint32_t indexOffset)
{
    utils::UnpackedVertex& uv0 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + 0]];
    utils::UnpackedVertex& uv1 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + 1]];
    utils::UnpackedVertex& uv2 = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + 2]];

    const float3 position0 = { uv0.position[0], uv0.position[1], uv0.position[2] };
    const float3 position1 = { uv1.position[0], uv1.position[1], uv1.position[2] };
    const float3 position2 = { uv2.position[0], uv2.position[1], uv2.position[2] };
    const float3 direction0 = position1 - position0;
    const float3 direction1 = position2 - position0;
    const float3 normal = Normalize(Cross(Normalize(direction1), Normalize(direction0)));

    const float3 axisX = float3(1.0f, 0.0f, 0.0f);
    const float3 axisY = float3(0.0f, 1.0f, 0.0f);
    const float3 bestAxis = abs(Dot33(axisX, normal)) < 0.9f ? axisX : axisY;
    const float3 tangent = Normalize(Cross(bestAxis, normal));

    for (size_t i = 0; i < 3; i++)
    {
        utils::UnpackedVertex& unpackedVertex = scene.unpackedVertices[vertexOffset + scene.indices[indexOffset + i]];
        unpackedVertex.normal[0] = normal.x;
        unpackedVertex.normal[1] = normal.y;
        unpackedVertex.normal[2] = normal.z;
        unpackedVertex.tangent[0] = tangent.x;
        unpackedVertex.tangent[1] = tangent.y;
        unpackedVertex.tangent[2] = tangent.z;

        utils::Vertex& vertex = scene.vertices[vertexOffset + scene.indices[indexOffset + i]];
        vertex.normal = Packed::uf4_to_uint<10, 10, 10, 2>(normal * 0.5f + 0.5f);
        vertex.tangent = Packed::uf4_to_uint<10, 10, 10, 2>(tangent * 0.5f + 0.5f);
    }
}

void Sample::GeneratePlane(utils::Scene& scene, float3 origin, float3 axisX, float3 axisY, float2 size, uint32_t subdivisionY, uint32_t vertexOffset, float uvScaling = 1.0f)
{
    const uint32_t subdivisionX = uint32_t(ceilf((size.x / size.y)) * subdivisionY);
    const float3 stepX = axisX * size.x / float(subdivisionX - 1);
    const float3 stepY = axisY * size.y / float(subdivisionY - 1);
    const float3 begin = origin - axisX * size.x * 0.5f - axisY * size.y * 0.5f;
    const float texCoordStepX = (size.x / (subdivisionX - 1)) * uvScaling;
    const float texCoordStepY = (size.y / (subdivisionY - 1)) * uvScaling;

    uint32_t baseVertex = (uint32_t)scene.vertices.size() - vertexOffset;

    float3 position;
    float2 texCoords;

    for (size_t i = 0; i < subdivisionY; i++)
    {
        for (size_t j = 0; j < subdivisionX; j++)
        {
            position = begin + stepX * (float)j + stepY * (float)i;
            texCoords = { texCoordStepX * (float)j, texCoordStepY * (float)i };
            PushVertex(scene, position.x, position.y, position.z, texCoords.x, texCoords.y);
        }
    }

    for (uint32_t i = 0; i < subdivisionY - 1; i++)
    {
        for (uint32_t j = 0; j < subdivisionX - 1; j++)
        {
            uint32_t pritimiveOffset = (uint32_t)scene.indices.size();
            scene.indices.push_back(utils::Index(baseVertex + i * subdivisionX + j));
            scene.indices.push_back(utils::Index(baseVertex + (i + 1) * subdivisionX + j));
            scene.indices.push_back(utils::Index(baseVertex + (i + 1) * subdivisionX + j + 1));
            ComputePrimitiveNormal(scene, vertexOffset, pritimiveOffset);
            scene.primitives.push_back({ CalculateWorldToUvUnits(scene, vertexOffset, pritimiveOffset) });

            pritimiveOffset = (uint32_t)scene.indices.size();
            scene.indices.push_back(utils::Index(baseVertex + (i + 1) * subdivisionX + j + 1));
            scene.indices.push_back(utils::Index(baseVertex + i * subdivisionX + j + 1));
            scene.indices.push_back(utils::Index(baseVertex + i * subdivisionX + j));
            ComputePrimitiveNormal(scene, vertexOffset, pritimiveOffset);
            scene.primitives.push_back({ CalculateWorldToUvUnits(scene, vertexOffset, pritimiveOffset) });
        }
    }
}

void Sample::GenerateGeometry(utils::Scene& scene)
{
    const uint32_t vertexOffset = (uint32_t)scene.unpackedVertices.size();
    const uint32_t indexOffset = (uint32_t)scene.indices.size();

    constexpr float cubeSize = 2.5f;
    const uint32_t subdivision = 2;
    constexpr float uvScaling = 1.0f;

    GeneratePlane(scene, float3(cubeSize * 0.5f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);
    GeneratePlane(scene, float3(-cubeSize * 0.5f, 0.0f, 0.0f), float3(0.0f, -1.0f, 0.0f), float3(0.0f, 0.0f, -1.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);
    GeneratePlane(scene, float3(0.0f, 0.0f, cubeSize * 0.5f), float3(0.0f, 1.0f, 0.0f), float3(1.0f, 0.0f, 0.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);
    GeneratePlane(scene, float3(0.0f, 0.0f, -cubeSize * 0.5f), float3(0.0f, -1.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);
    GeneratePlane(scene, float3(0.0f, cubeSize * 0.5f, 0.0f), float3(1.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);
    GeneratePlane(scene, float3(0.0f, -cubeSize * 0.5f, 0.0f), float3(1.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), float2(cubeSize), subdivision, vertexOffset, uvScaling);

    const uint32_t indexNum = (uint32_t)scene.indices.size() - indexOffset;
    const uint32_t vertexNum = (uint32_t)scene.vertices.size() - vertexOffset;

    utils::Mesh mesh = {};
    mesh.aabb.vMin = { -100.0f, -1.0f,    0.0f };
    mesh.aabb.vMax = { 100.0f, -1.0f,  100.0f };
    mesh.indexNum = indexNum;
    mesh.indexOffset = indexOffset;
    mesh.vertexOffset = vertexOffset;
    mesh.vertexNum = vertexNum;

    const uint32_t meshIndex = (uint32_t)scene.meshes.size();

    uint32_t diffuseMapIndex;
    uint32_t normalMapIndex;
    uint32_t specularMapIndex;
    {
        utils::Texture* texture = new utils::Texture;
        const std::string& texPath = utils::GetFullPath("fence_albedo.png", utils::DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture, true));
        diffuseMapIndex = (uint32_t)scene.textures.size();
        scene.textures.push_back(texture);
    }
    scene.meshes.push_back(mesh);
    {
        utils::Texture* texture = new utils::Texture;
        const std::string& texPath = utils::GetFullPath("fence_normal.png", utils::DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture, true));
        normalMapIndex = (uint32_t)scene.textures.size();
        scene.textures.push_back(texture);
    }
    {
        utils::Texture* texture = new utils::Texture;
        const std::string& texPath = utils::GetFullPath("fence_specular.png", utils::DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture, true));
        specularMapIndex = (uint32_t)scene.textures.size();
        scene.textures.push_back(texture);
    }

    if (scene.materialsGroups.size() == 1)
    {
        utils::MaterialGroup materialGroup = scene.materialsGroups[0];
        materialGroup.materialOffset = materialGroup.materialOffset + materialGroup.materialNum;
        materialGroup.materialNum = 0;

        scene.materialsGroups.push_back(materialGroup);
    }
    else
    {
        for (size_t i = 0; i < scene.instances.size(); i++)
        {
            if (scene.instances[i].materialIndex >= scene.materialsGroups[1].materialOffset)
                scene.instances[i].materialIndex++;
        }
    }

    utils::Material newMaterial = {};
    newMaterial.avgBaseColor = scene.textures[diffuseMapIndex]->avgColor;
    newMaterial.avgSpecularColor = scene.textures[specularMapIndex]->avgColor;
    newMaterial.diffuseMapIndex = diffuseMapIndex;
    newMaterial.normalMapIndex = normalMapIndex;
    newMaterial.specularMapIndex = specularMapIndex;
    newMaterial.alphaMode = utils::AlphaMode::PREMULTIPLIED;

    const uint32_t materialIndex = scene.materialsGroups[1].materialOffset;

    scene.materialsGroups[1].materialNum++;
    scene.materials.insert(scene.materials.begin() + scene.materialsGroups[1].materialOffset, newMaterial);

    for (size_t i = 0; i < 10; i++)
    {
        for (size_t j = 0; j < 10; j++)
        {
            utils::Instance instance = {};
            instance.materialIndex = materialIndex;
            instance.meshIndex = meshIndex;
            instance.rotation = float4x4::Identity();
            instance.rotationPrev = instance.rotation;
            instance.position = { -cubeSize * 2.0f * 5 + cubeSize * 2.0f * i, -cubeSize * 2.0f * 5 + cubeSize * 2.0f * j, 3.0 };
            instance.positionPrev = instance.position;

            scene.instances.push_back(instance);
        }
    }
}

void Sample::PrepareFrame(uint32_t frameIndex)
{
    m_ForceHistoryReset = false;
    m_PrevSettings = m_Settings;
    m_Camera.SavePreviousState();

    PrepareUserInterface();

    if (IsKeyToggled(Key::Tab))
        m_ShowUi = !m_ShowUi;
    if (IsKeyToggled(Key::F1))
        m_Settings.debug = Step(0.5f, 1.0f - m_Settings.debug);
    if (IsKeyToggled(Key::F3))
        m_Settings.emission = !m_Settings.emission;
    if (IsKeyToggled(Key::Space))
        m_Settings.pauseAnimation = !m_Settings.pauseAnimation;
    if (IsKeyToggled(Key::PageDown) || IsKeyToggled(Key::Num3))
    {
        m_Settings.denoiser++;
        if (m_Settings.denoiser > DENOISER_MAX_NUM - 1)
            m_Settings.denoiser = 0;
    }
    if (IsKeyToggled(Key::PageUp) || IsKeyToggled(Key::Num9))
    {
        m_Settings.denoiser--;
        if (m_Settings.denoiser < 0)
            m_Settings.denoiser = DENOISER_MAX_NUM - 1;
    }

    if (!IsKeyPressed(Key::LAlt) && m_ShowUi)
    {
        ImGui::SetNextWindowPos(ImVec2(m_Settings.windowAlignment ? 5.0f : GetOutputResolution().x - m_UiWidth - 5.0f, 5.0f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f));
        ImGui::Begin("Settings [Tab]", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        {
            float avgFrameTime = m_Timer.GetVerySmoothedFrameTime();

            char buf[256];
            snprintf(buf, sizeof(buf), "%.1f FPS (%.2f ms)", 1000.0f / avgFrameTime, avgFrameTime);

            ImVec4 colorFps = UI_GREEN;
            if (avgFrameTime > 1000.0f / 59.5f)
                colorFps = UI_YELLOW;
            if (avgFrameTime > 1000.0f / 29.5f)
                colorFps = UI_RED;

            float lo = avgFrameTime * 0.5f;
            float hi = avgFrameTime * 1.5f;

            const uint32_t N = helper::GetCountOf(m_FrameTimes);
            uint32_t head = frameIndex % N;
            m_FrameTimes[head] = m_Timer.GetFrameTime();
            ImGui::PushStyleColor(ImGuiCol_Text, colorFps);
            ImGui::PlotLines("", m_FrameTimes.data(), N, head, buf, lo, hi, ImVec2(0.0f, 70.0f));
            ImGui::PopStyleColor();

            ImGui::PushID("Profiler0");
            {
                ImGui::Separator();
                ImGui::BeginTable("", 2, ImGuiTabBarFlags_::ImGuiTabBarFlags_None);
                size_t eventsCount = 0;
                const ProfilerEvent* events = m_Profiler.GetPerformanceEvents(eventsCount);
                for (size_t i = 0; i < eventsCount; ++i)
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text(events[i].name.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%6.3f(ms)", events[i].GetSmootherDelta());
                }
                ImGui::EndTable();
                ImGui::Separator();
            }
            ImGui::PopID();
            m_UiWidth = ImGui::GetWindowWidth();

            if (IsButtonPressed(Button::Right))
            {
                ImGui::Text("Move - W/S/A/D");
                ImGui::Text("Accelerate - MOUSE SCROLL");
            }
            else
            {
                ImGui::Checkbox("Show all settings", &m_ShowFullSettings);
                AppendOmmImguiSettings();
                if (m_ShowFullSettings)
                {
                    // "Camera" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    bool isUnfolded = ImGui::CollapsingHeader("CAMERA (press RIGHT MOUSE BOTTON for free-fly mode)", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("CAMERA");
                    if (isUnfolded)
                    {
                        static const char* onScreenModes[] =
                        {
                    #if( NRD_MODE == OCCLUSION )
                            "Diffuse occlusion",
                            "Specular occlusion",
                    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                            "Diffuse occlusion",
                    #else
                            "Final",
                            "Denoised diffuse",
                            "Denoised specular",
                            "Diffuse occlusion",
                            "Specular occlusion",
                            "Shadow",
                            "Base color",
                            "Normal",
                            "Roughness",
                            "Metalness",
                            "World units",
                            "Mesh",
                            "Mip level (primary)",
                            "Mip level (specular)",
                    #endif
                        };

                        static const char* motionMode[] =
                        {
                            "Left / Right",
                            "Up / Down",
                            "Forward / Backward",
                            "Mixed",
                            "Pan",
                        };

                        static const char* mvType[] =
                        {
                            "2D",
                            "2.5D",
                            "3D",
                        };

                        ImGui::SliderFloat("FOV (deg)", &m_Settings.camFov, 1.0f, 160.0f, "%.1f");
                        ImGui::SliderFloat("Exposure", &m_Settings.exposure, 0.0f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                        ImGui::SliderFloat("Resolution scale (%)", &m_Settings.resolutionScale, m_MinResolutionScale, 1.0f, "%.3f");
                        ImGui::Combo("On screen", &m_Settings.onScreen, onScreenModes, helper::GetCountOf(onScreenModes));
                        ImGui::Checkbox("Ortho", &m_Settings.ortho);
                        ImGui::SameLine();
                        ImGui::Checkbox("FPS cap", &m_Settings.limitFps);
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, (!m_Settings.cameraJitter && (m_Settings.TAA || m_Settings.DLSS)) ? UI_RED : UI_DEFAULT);
                        ImGui::Checkbox("Jitter", &m_Settings.cameraJitter);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.animatedObjects && !m_Settings.pauseAnimation && m_Settings.mvType == MV_2D) ? UI_RED : UI_DEFAULT);
                        ImGui::Combo("MV", &m_Settings.mvType, mvType, helper::GetCountOf(mvType));
                        ImGui::PopStyleColor();

                        ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.motionStartTime > 0.0 ? UI_YELLOW : UI_DEFAULT);
                        bool isPressed = ImGui::Button("Animation");
                        ImGui::PopStyleColor();
                        if (isPressed)
                            m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? 0.0 : -1.0;
                        if (m_Settings.motionStartTime > 0.0)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("Linear", &m_Settings.linearMotion);
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                            ImGui::Combo("Mode", &m_Settings.motionMode, motionMode, helper::GetCountOf(motionMode));
                            ImGui::SliderFloat("Slower / Faster", &m_Settings.emulateMotionSpeed, -10.0f, 10.0f);
                        }

                        if (m_Settings.limitFps)
                            ImGui::SliderFloat("Min / Max FPS", &m_Settings.maxFps, 30.0f, 120.0f, "%.0f");
                    }
                    ImGui::PopID();

                    // "Antialiasing, upscaling & sharpening" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("ANTIALIASING, UPSCALING & SHARPENING", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("ANTIALIASING");
                    if (isUnfolded)
                    {
                        if (m_DLSS.IsInitialized())
                        {
                            ImGui::Checkbox("DLSS", &m_Settings.DLSS);
                            ImGui::SameLine();
                        }
                        if (!m_Settings.DLSS)
                        {
                            ImGui::PushStyleColor(ImGuiCol_Text, (m_ReblurSettings.enableReferenceAccumulation && m_Settings.TAA) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("TAA", &m_Settings.TAA);
                            ImGui::PopStyleColor();
                            ImGui::SameLine();
                        }
                        bool isNis = m_Settings.NIS && m_Settings.separator == 0.0f;
                        if (!m_Settings.DLSS)
                            ImGui::Checkbox("NIS", &m_Settings.NIS);
                        if (isNis || m_Settings.DLSS)
                        {
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                            ImGui::SliderFloat(m_Settings.DLSS ? "Sharpness" : "Sharpness", &m_Settings.sharpness, 0.0f, 1.0f, "%.2f");
                        }
                    }
                    ImGui::PopID();

                    // "Materials" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("MATERIALS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("MATERIALS");
                    if (isUnfolded)
                    {
                        static const char* forcedMaterial[] =
                        {
                            "None",
                            "Gypsum",
                            "Cobalt",
                        };

                        ImGui::SliderFloat2("Roughness / Metalness", &m_Settings.roughnessOverride, 0.0f, 1.0f, "%.3f");
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.emissiveObjects && !m_Settings.emission) ? UI_YELLOW : UI_DEFAULT);
                        ImGui::Checkbox("Emission [F3]", &m_Settings.emission);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x);
                        ImGui::Combo("Material", &m_Settings.forcedMaterial, forcedMaterial, helper::GetCountOf(forcedMaterial));
                        if (m_Settings.emission)
                            ImGui::SliderFloat("Emission intensity", &m_Settings.emissionIntensity, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    }
                    ImGui::PopID();

                    if (m_Settings.onScreen == 10)
                        ImGui::SliderFloat("Units in 1 meter", &m_Settings.meterToUnitsMultiplier, 0.001f, 100.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
                    else
                    {
                        // "World" section
                        snprintf(buf, sizeof(buf) - 1, "WORLD%s", (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera) ? (m_Settings.pauseAnimation ? " (SPACE - unpause)" : " (SPACE - pause)") : "");

                        ImGui::NewLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                        ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                        isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();

                        ImGui::PushID("WORLD");
                        if (isUnfolded)
                        {
                            ImGui::Checkbox("Animate sun", &m_Settings.animateSun);
                            if (!m_Scene.animations.empty() && m_Scene.animations[m_Settings.activeAnimation].cameraNode.HasAnimation())
                            {
                                ImGui::SameLine();
                                ImGui::Checkbox("Animate camera", &m_Settings.animateCamera);
                            }
                            if (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera)
                            {
                                ImGui::SameLine();
                                ImGui::Checkbox("Pause", &m_Settings.pauseAnimation);
                            }

                            ImGui::SliderFloat2("Sun position (deg)", &m_Settings.sunAzimuth, -180.0f, 180.0f, "%.2f");
                            ImGui::SliderFloat("Sun size (deg)", &m_Settings.sunAngularDiameter, 0.0f, 3.0f, "%.1f");
                            if (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera)
                                ImGui::SliderFloat("Slower / Faster", &m_Settings.animationSpeed, -10.0f, 10.0f);

                            ImGui::Checkbox("Objects", &m_Settings.animatedObjects);
                            if (m_Settings.animatedObjects)
                            {
                                ImGui::SameLine();
                                ImGui::Checkbox("9", &m_Settings.nineBrothers);
                                ImGui::SameLine();
                                ImGui::Checkbox("Blink", &m_Settings.blink);
                                ImGui::SameLine();
                                ImGui::Checkbox("Emissive", &m_Settings.emissiveObjects);
                                if (!m_Settings.nineBrothers)
                                    ImGui::SliderInt("Object number", &m_Settings.animatedObjectNum, 1, (int32_t)MAX_ANIMATED_INSTANCE_NUM);
                                ImGui::SliderFloat("Object scale", &m_Settings.animatedObjectScale, 0.1f, 2.0f);
                            }

                            if (m_Settings.animateCamera)
                            {
                                if (m_Scene.animations[m_Settings.activeAnimation].durationMs != 0.0f)
                                {
                                    char animationLabel[128];
                                    snprintf(animationLabel, sizeof(animationLabel), "Animation %.1f sec (%%)", 0.001f * m_Scene.animations[m_Settings.activeAnimation].durationMs / (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed)));
                                    ImGui::SliderFloat(animationLabel, &m_Settings.animationProgress, 0.0f, 99.999f);

                                    if (m_Scene.animations.size() > 1)
                                    {
                                        char items[1024] = { '\0' };
                                        size_t offset = 0;
                                        char* iterator = items;
                                        for (auto animation : m_Scene.animations)
                                        {
                                            const size_t size = std::min(sizeof(items), animation.animationName.length() + 1);
                                            memcpy(iterator + offset, animation.animationName.c_str(), size);
                                            offset += animation.animationName.length() + 1;
                                        }
                                        ImGui::Combo("Animated scene", &m_Settings.activeAnimation, items, helper::GetCountOf(m_Scene.animations));
                                    }
                                }
                            }
                        }
                        ImGui::PopID();

                        // "Indirect rays" section
                        ImGui::NewLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                        ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                        isUnfolded = ImGui::CollapsingHeader("INDIRECT RAYS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();

                        ImGui::PushID("INDIRECT RAYS");
                        if (isUnfolded)
                        {
                            const float sceneRadiusInMeters = m_Scene.aabb.GetRadius() / m_Settings.meterToUnitsMultiplier;

                            static const char* resolution[] =
                            {
                                "Full",
                                "Full (probabilistic)",
                                "Half",
                            };

#if( NRD_MODE == NORMAL || NRD_MODE == SH )
                            ImGui::SliderInt2("Samples / Bounces", &m_Settings.rpp, 1, 8);
#else
                            ImGui::SliderInt("Samples", &m_Settings.rpp, 1, 8);
#endif
                            ImGui::SliderFloat("AO / SO range (m)", &m_Settings.hitDistScale, 0.01f, sceneRadiusInMeters, "%.2f");
                            ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.denoiser == REFERENCE && m_Settings.tracingMode > RESOLUTION_FULL_PROBABILISTIC) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Combo("Resolution", &m_Settings.tracingMode, resolution, helper::GetCountOf(resolution));
                            ImGui::PopStyleColor();

                            ImGui::Checkbox("Diffuse", &m_Settings.indirectDiffuse);
                            ImGui::SameLine();
                            ImGui::Checkbox("Specular", &m_Settings.indirectSpecular);
                            ImGui::SameLine();
                            ImGui::Checkbox("Trim lobe", &m_Settings.specularLobeTrimming);
                            ImGui::SameLine();
                            ImGui::Checkbox("Normal map", &m_Settings.normalMap);

#if( NRD_MODE == NORMAL || NRD_MODE == SH )
                            const float3& sunDirection = GetSunDirection();
                            bool cmp = sunDirection.z < 0.0f && m_Settings.importanceSampling;
                            if (cmp)
                                ImGui::PushStyleColor(ImGuiCol_Text, UI_RED);
                            ImGui::Checkbox("IS", &m_Settings.importanceSampling);
                            if (cmp)
                                ImGui::PopStyleColor();

                            ImGui::SameLine();
                            ImGui::Checkbox("Use prev frame", &m_Settings.usePrevFrame);

                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.ambient && m_Settings.denoiser == RELAX) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("Ambient", &m_Settings.ambient);
                            ImGui::PopStyleColor();

                            if (m_Settings.tracingMode != RESOLUTION_HALF)
                            {
                                ImGui::SameLine();
                                ImGui::Checkbox("PSR", &m_Settings.PSR);
                            }
#else
                            if (m_Settings.tracingMode != RESOLUTION_HALF)
                                ImGui::Checkbox("PSR", &m_Settings.PSR);
#endif
                        }
                        ImGui::PopID();

                        // "NRD" section
                        static const char* denoiser[] =
                        {
                        #if( NRD_MODE == OCCLUSION )
                            "REBLUR_OCCLUSION",
                        #elif( NRD_MODE == SH )
                            "REBLUR_SH + SIGMA",
                        #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                            "REBLUR_DIRECTIONAL_OCCLUSION",
                        #else
                            "REBLUR + SIGMA",
                        #endif
                            "RELAX + SIGMA",
                            "REFERENCE",
                        };
                        const nrd::LibraryDesc& nrdLibraryDesc = nrd::GetLibraryDesc();
                        snprintf(buf, sizeof(buf) - 1, "NRD v%u.%u.%u - %s [PgDown / PgUp]", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild, denoiser[m_Settings.denoiser]);

                        ImGui::NewLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                        ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                        isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();

                        ImGui::PushID("NRD");
                        if (isUnfolded)
                        {
                            static const char* hitDistanceReconstructionMode[] =
                            {
                                "Off",
                                "3x3",
                                "5x5",
                            };

                            if (m_DebugNRD)
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, m_ShowValidationOverlay ? UI_YELLOW : UI_DEFAULT);
                                ImGui::Checkbox("Validation overlay", &m_ShowValidationOverlay);
                                ImGui::PopStyleColor();
                            }

                            if (ImGui::Button("<<"))
                            {
                                m_Settings.denoiser--;
                                if (m_Settings.denoiser < 0)
                                    m_Settings.denoiser = DENOISER_MAX_NUM - 1;
                            }

                            ImGui::SameLine();
                            if (ImGui::Button(">>"))
                            {
                                m_Settings.denoiser++;
                                if (m_Settings.denoiser > DENOISER_MAX_NUM - 1)
                                    m_Settings.denoiser = 0;
                            }

                            ImGui::SameLine();
                            m_ForceHistoryReset = ImGui::Button("Reset");

                            if (m_Settings.denoiser == REBLUR)
                            {
                                nrd::ReblurSettings defaults = {};

                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                {
                                    defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                    defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                                }

                                bool isSame = true;
                                if (m_ReblurSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                    isSame = false;
                                else if (m_ReblurSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                    isSame = false;
                                else if (m_ReblurSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                    isSame = false;
                                else if (m_ReblurSettings.blurRadius != defaults.blurRadius)
                                    isSame = false;
                                else if (m_ReblurSettings.historyFixStrideBetweenSamples != defaults.historyFixStrideBetweenSamples)
                                    isSame = false;
                                else if (m_ReblurSettings.lobeAngleFraction != defaults.lobeAngleFraction)
                                    isSame = false;
                                else if (m_ReblurSettings.roughnessFraction != defaults.roughnessFraction)
                                    isSame = false;
                                else if (m_ReblurSettings.responsiveAccumulationRoughnessThreshold != defaults.responsiveAccumulationRoughnessThreshold)
                                    isSame = false;
                                else if (m_ReblurSettings.stabilizationStrength != defaults.stabilizationStrength)
                                    isSame = false;
                                else if (m_ReblurSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                    isSame = false;
                                else if (m_ReblurSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                    isSame = false;
                                else if (m_ReblurSettings.enableReferenceAccumulation != defaults.enableReferenceAccumulation)
                                    isSame = false;
                                else if (m_ReblurSettings.enablePerformanceMode != defaults.enablePerformanceMode)
                                    isSame = false;
                                else if (m_ReblurSettings.antilagHitDistanceSettings.enable != true)
                                    isSame = false;
                                else if (m_ReblurSettings.antilagIntensitySettings.enable != true)
                                    isSame = false;

                                ImGui::SameLine();
                                if (ImGui::Button("No spatial"))
                                {
                                    m_ReblurSettings.blurRadius = 0.0f;
                                    m_ReblurSettings.diffusePrepassBlurRadius = 0.0f;
                                    m_ReblurSettings.specularPrepassBlurRadius = 0.0f;
                                    m_ReblurSettings.antilagHitDistanceSettings.enable = false;
                                    m_ReblurSettings.antilagIntensitySettings.enable = false;
                                }

                                ImGui::SameLine();
                                if (ImGui::Button(m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum ? "No fast" : "Fast"))
                                {
                                    if (m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum)
                                        m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                    else
                                        m_Settings.maxFastAccumulatedFrameNum = defaults.maxFastAccumulatedFrameNum;
                                }

                                ImGui::SameLine();
                                ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                                if (ImGui::Button("Defaults"))
                                {
                                    m_ReblurSettings = defaults;
                                    m_ReblurSettings.antilagIntensitySettings.enable = true;
                                }
                                ImGui::PopStyleColor();

                                ImGui::Checkbox("Adaptive radius", &m_Settings.adaptRadiusToResolution);
                                ImGui::SameLine();
                                ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);

                                ImGui::Checkbox("Ref accum", &m_ReblurSettings.enableReferenceAccumulation);
                                ImGui::SameLine();
                                ImGui::Checkbox("Anti-firefly", &m_ReblurSettings.enableAntiFirefly);
                                ImGui::SameLine();
                                ImGui::Checkbox("Perf mode", &m_ReblurSettings.enablePerformanceMode);
#if( NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION )
                                ImGui::SameLine();
                                ImGui::Checkbox("Resolve", &m_Resolve);
#endif

                                ImGui::SliderFloat("Disocclusion (%)", &m_Settings.disocclusionThreshold, 0.25f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                                ImGui::SliderInt2("History length (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d", ImGuiSliderFlags_Logarithmic);

                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                {
                                    ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                    {
                                        int32_t v = (int32_t)m_ReblurSettings.hitDistanceReconstructionMode;
                                        ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                        m_ReblurSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                    }
                                    ImGui::PopStyleColor();
                                }

#if( NRD_MODE != OCCLUSION )
                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                    ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.diffusePrepassBlurRadius != 0.0f && m_ReblurSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                                ImGui::SliderFloat2("Pre-pass radius (px)", &m_ReblurSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                    ImGui::PopStyleColor();
#endif

                                ImGui::SliderFloat("Blur base radius (px)", &m_ReblurSettings.blurRadius, 0.0f, 60.0f, "%.1f");
                                ImGui::SliderFloat("Lobe fraction", &m_ReblurSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                                ImGui::SliderFloat("Roughness fraction", &m_ReblurSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                                ImGui::SliderFloat("History fix stride", &m_ReblurSettings.historyFixStrideBetweenSamples, 0.0f, 20.0f, "%.1f");
                                ImGui::SliderInt("History fix frames", (int32_t*)&m_ReblurSettings.historyFixFrameNum, 0, 6);
#if( NRD_MODE != OCCLUSION )
                                ImGui::SliderFloat("Stabilization (%)", &m_ReblurSettings.stabilizationStrength, 0.0f, 1.0f, "%.2f");
#endif
                                ImGui::SetNextItemWidth(ImGui::CalcItemWidth() * 0.5f);
                                ImGui::SliderFloat("Responsive accumulation roughness threshold", &m_ReblurSettings.responsiveAccumulationRoughnessThreshold, 0.0f, 1.0f, "%.2f");

#if( NRD_MODE != OCCLUSION )
                                if (m_ReblurSettings.stabilizationStrength != 0.0f)
                                {
                                    ImGui::Text("ANTI-LAG:");
                                    ImGui::Checkbox("Intensity", &m_ReblurSettings.antilagIntensitySettings.enable);
                                    ImGui::SameLine();
                                    ImGui::Text("[%.1f%%; %.1f%%; %.1f]", m_ReblurSettings.antilagIntensitySettings.thresholdMin * 100.0, m_ReblurSettings.antilagIntensitySettings.thresholdMax * 100.0, m_ReblurSettings.antilagIntensitySettings.sigmaScale);

                                    ImGui::SameLine();
                                    ImGui::Checkbox("Hit dist", &m_ReblurSettings.antilagHitDistanceSettings.enable);
                                    ImGui::SameLine();
                                    ImGui::Text("[%.1f%%; %.1f%%; %.1f]", m_ReblurSettings.antilagHitDistanceSettings.thresholdMin * 100.0, m_ReblurSettings.antilagHitDistanceSettings.thresholdMax * 100.0, m_ReblurSettings.antilagHitDistanceSettings.sigmaScale);
                                }
#endif
                            }
                            else if (m_Settings.denoiser == RELAX)
                            {
                                nrd::RelaxDiffuseSpecularSettings defaults = {};

                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                {
                                    defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                    defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                                }

                                bool isSame = true;
                                if (m_RelaxSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                    isSame = false;
                                else if (m_RelaxSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                    isSame = false;
                                else if (m_RelaxSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                    isSame = false;
                                else if (m_RelaxSettings.diffusePhiLuminance != defaults.diffusePhiLuminance)
                                    isSame = false;
                                else if (m_RelaxSettings.specularPhiLuminance != defaults.specularPhiLuminance)
                                    isSame = false;
                                else if (m_RelaxSettings.diffuseLobeAngleFraction != defaults.diffuseLobeAngleFraction)
                                    isSame = false;
                                else if (m_RelaxSettings.specularLobeAngleFraction != defaults.specularLobeAngleFraction)
                                    isSame = false;
                                else if (m_RelaxSettings.specularVarianceBoost != defaults.specularVarianceBoost)
                                    isSame = false;
                                else if (m_RelaxSettings.specularLobeAngleSlack != defaults.specularLobeAngleSlack)
                                    isSame = false;
                                else if (m_RelaxSettings.historyFixEdgeStoppingNormalPower != defaults.historyFixEdgeStoppingNormalPower)
                                    isSame = false;
                                else if (m_RelaxSettings.historyFixStrideBetweenSamples != defaults.historyFixStrideBetweenSamples)
                                    isSame = false;
                                else if (m_RelaxSettings.historyClampingColorBoxSigmaScale != defaults.historyClampingColorBoxSigmaScale)
                                    isSame = false;
                                else if (m_RelaxSettings.spatialVarianceEstimationHistoryThreshold != defaults.spatialVarianceEstimationHistoryThreshold)
                                    isSame = false;
                                else if (m_RelaxSettings.atrousIterationNum != defaults.atrousIterationNum)
                                    isSame = false;
                                else if (m_RelaxSettings.diffuseMinLuminanceWeight != defaults.diffuseMinLuminanceWeight)
                                    isSame = false;
                                else if (m_RelaxSettings.specularMinLuminanceWeight != defaults.specularMinLuminanceWeight)
                                    isSame = false;
                                else if (m_RelaxSettings.depthThreshold != defaults.depthThreshold)
                                    isSame = false;
                                else if (m_RelaxSettings.roughnessFraction != defaults.roughnessFraction)
                                    isSame = false;
                                else if (m_RelaxSettings.roughnessEdgeStoppingRelaxation != defaults.roughnessEdgeStoppingRelaxation)
                                    isSame = false;
                                else if (m_RelaxSettings.normalEdgeStoppingRelaxation != defaults.normalEdgeStoppingRelaxation)
                                    isSame = false;
                                else if (m_RelaxSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                    isSame = false;
                                else if (m_RelaxSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                    isSame = false;
                                else if (m_RelaxSettings.enableRoughnessEdgeStopping != defaults.enableRoughnessEdgeStopping)
                                    isSame = false;

                                ImGui::SameLine();
                                if (ImGui::Button("No spatial"))
                                {
                                    m_RelaxSettings.diffusePhiLuminance = 0.0f;
                                    m_RelaxSettings.specularPhiLuminance = 0.0f;
                                    m_RelaxSettings.diffusePrepassBlurRadius = 0.0f;
                                    m_RelaxSettings.specularPrepassBlurRadius = 0.0f;
                                    m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = 0;
                                }

                                ImGui::SameLine();
                                if (ImGui::Button(m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum ? "No fast" : "Fast"))
                                {
                                    if (m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum)
                                        m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                    else
                                        m_Settings.maxFastAccumulatedFrameNum = defaults.diffuseMaxFastAccumulatedFrameNum;
                                }

                                ImGui::SameLine();
                                ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                                if (ImGui::Button("Defaults"))
                                    m_RelaxSettings = defaults;
                                ImGui::PopStyleColor();

                                ImGui::Checkbox("Adaptive radius", &m_Settings.adaptRadiusToResolution);
                                ImGui::SameLine();
                                ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);

                                ImGui::Checkbox("Roughness edge stopping", &m_RelaxSettings.enableRoughnessEdgeStopping);
                                ImGui::SameLine();
                                ImGui::Checkbox("Anti-firefly", &m_RelaxSettings.enableAntiFirefly);
                        #if( NRD_MODE == SH)
                            ImGui::SameLine();
                            ImGui::Checkbox("Resolve", &m_Resolve);
                        #endif

                                ImGui::SliderFloat("Disocclusion (%)", &m_Settings.disocclusionThreshold, 0.25f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                                ImGui::SliderInt2("History length (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d", ImGuiSliderFlags_Logarithmic);

                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                {
                                    ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                    {
                                        int32_t v = (int32_t)m_RelaxSettings.hitDistanceReconstructionMode;
                                        ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                        m_RelaxSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                    }
                                    ImGui::PopStyleColor();
                                }

#if( NRD_MODE != OCCLUSION )
                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                    ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.diffusePrepassBlurRadius != 0.0f && m_RelaxSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                                ImGui::SliderFloat2("Pre-pass radius (px)", &m_RelaxSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                                if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                    ImGui::PopStyleColor();
#endif

                                ImGui::SliderInt("A-trous iterations", (int32_t*)&m_RelaxSettings.atrousIterationNum, 2, 8);
                                ImGui::SliderFloat2("Diff-Spec luma weight", &m_RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f, "%.1f");
                                ImGui::SliderFloat2("Min luma weight", &m_RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f, "%.2f");
                                ImGui::SliderFloat("Spec lobe angle slack", &m_RelaxSettings.specularLobeAngleSlack, 0.0f, 89.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                                ImGui::SliderFloat("Depth threshold", &m_RelaxSettings.depthThreshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                                ImGui::Text("Diffuse lobe / Specular lobe / Roughness:");
                                ImGui::SliderFloat3("Fraction", &m_RelaxSettings.diffuseLobeAngleFraction, 0.0f, 1.0f, "%.2f");
                                ImGui::Text("Luminance / Normal / Roughness:");
                                ImGui::SliderFloat3("Relaxation", &m_RelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f, "%.2f");
                                ImGui::SliderFloat("Spec variance boost", &m_RelaxSettings.specularVarianceBoost, 0.0f, 8.0f, "%.2f");
                                ImGui::SliderFloat("Clamping sigma scale", &m_RelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 10.0f, "%.1f");
                                ImGui::SliderInt("History threshold", (int32_t*)&m_RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, 10);

                                ImGui::Text("HISTORY FIX:");
                                ImGui::SliderFloat("Normal weight power", &m_RelaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 128.0f, "%.1f");
                                ImGui::SliderFloat("Stride", &m_RelaxSettings.historyFixStrideBetweenSamples, 0.0f, 20.0f, "%.1f");
                                ImGui::SliderInt("Frames", (int32_t*)&m_RelaxSettings.historyFixFrameNum, 0, 6);
                            }
                            else if (m_Settings.denoiser == REFERENCE)
                            {
                                float t = (float)m_ReferenceSettings.maxAccumulatedFrameNum;
                                ImGui::SliderFloat("History length (frames)", &t, 0.0f, 1024.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                                m_ReferenceSettings.maxAccumulatedFrameNum = (int32_t)t;
                            }
                        }
                        ImGui::PopID();

                        // "Other" section
                        ImGui::NewLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                        ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                        isUnfolded = ImGui::CollapsingHeader("OTHER", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();

                        ImGui::PushID("OTHER");
                        if (isUnfolded)
                        {
                            ImGui::SliderFloat("Debug [F1]", &m_Settings.debug, 0.0f, 1.0f, "%.6f");
                            ImGui::SliderFloat("Input / Denoised", &m_Settings.separator, 0.0f, 1.0f, "%.2f");

                            if (ImGui::Button(m_Settings.windowAlignment ? ">>" : "<<"))
                                m_Settings.windowAlignment = !m_Settings.windowAlignment;

                            ImGui::SameLine();
                            if (ImGui::Button("Reload shaders"))
                            {
                                CreatePipelines();
                                printf("Ready!\n");
                            }

                            ImGui::SameLine();
                            if (ImGui::Button("Defaults"))
                            {
                                m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                                m_Settings = m_DefaultSettings;
                                m_RelaxSettings = {};
                                m_ReblurSettings = {};
                                m_ReblurSettings.antilagIntensitySettings.enable = true;
                                m_ForceHistoryReset = true;
                            }
                        }
                        ImGui::PopID();

                        // "Tests" section
                        ImGui::NewLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                        ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                        isUnfolded = ImGui::CollapsingHeader("TESTS [F2]", ImGuiTreeNodeFlags_CollapsingHeader);
                        ImGui::PopStyleColor();
                        ImGui::PopStyleColor();

                        ImGui::PushID("TESTS");
                        if (isUnfolded)
                        {
                            float buttonWidth = 25.0f * float(GetWindowResolution().x) / float(GetOutputResolution().x);

                            char s[64];
                            std::string sceneName = std::string(utils::GetFileName(m_SceneFile));
                            size_t dotPos = sceneName.find_last_of(".");
                            if (dotPos != std::string::npos)
                            sceneName = sceneName.substr(0, dotPos) + ".bin";
                            const std::string path = utils::GetFullPath(sceneName, utils::DataFolder::TESTS);
                            const uint32_t testByteSize = sizeof(m_Settings) + Camera::GetStateSize();

                            // Get number of tests
                            if (m_TestNum == uint32_t(-1))
                            {
                                FILE* fp = fopen(path.c_str(), "rb");
                                if (fp)
                                {
                                    // Use this code to convert tests to reflect new Settings and Camera layouts
#if 0
                                    typedef Settings SettingsOld; // adjust if needed
                                    typedef Camera CameraOld; // adjust if needed

                                    const uint32_t oldItemSize = sizeof(SettingsOld) + CameraOld::GetStateSize();

                                    fseek(fp, 0, SEEK_END);
                                    m_TestNum = ftell(fp) / oldItemSize;
                                    fseek(fp, 0, SEEK_SET);

                                    FILE* fpNew;
                                    fopen_s(&fpNew, (path + ".new").c_str(), "wb");

                                    for (uint32_t i = 0; i < m_TestNum && fpNew; i++)
                                    {
                                        SettingsOld settingsOld;
                                        fread_s(&settingsOld, sizeof(SettingsOld), 1, sizeof(SettingsOld), fp);

                                        CameraOld cameraOld;
                                        fread_s(cameraOld.GetState(), CameraOld::GetStateSize(), 1, CameraOld::GetStateSize(), fp);

                                        // Convert Old to New here
                                        m_Settings = settingsOld;
                                        m_Camera.state = cameraOld.state;

                                        // ...

                                        fwrite(&m_Settings, 1, sizeof(m_Settings), fpNew);
                                        fwrite(m_Camera.GetState(), 1, Camera::GetStateSize(), fpNew);
                                    }

                                    fclose(fp);
                                    fclose(fpNew);

                                    __debugbreak();
#endif

                                    fseek(fp, 0, SEEK_END);
                                    m_TestNum = ftell(fp) / testByteSize;
                                    fclose(fp);
                                }
                                else
                                    m_TestNum = 0;
                            }

                            // Adjust current test index
                            bool isTestChanged = false;
                            if (IsKeyToggled(Key::F2) && m_TestNum)
                            {
                                m_LastSelectedTest++;
                                isTestChanged = true;
                            }

                            if (m_LastSelectedTest == uint32_t(-1) || !m_TestNum)
                                m_LastSelectedTest = uint32_t(-1);
                            else
                                m_LastSelectedTest %= m_TestNum;

                            // Main buttons
                            uint32_t i = 0;
                            for (; i < m_TestNum; i++)
                            {
                                snprintf(s, sizeof(s), "%u", i + 1);

                                if (i % 14 != 0)
                                    ImGui::SameLine();

                                bool isColorChanged = false;
                                if (m_improveMeTests && std::find(m_improveMeTests->begin(), m_improveMeTests->end(), i + 1) != m_improveMeTests->end())
                                {
                                    ImGui::PushStyleColor(ImGuiCol_Text, UI_RED);
                                    isColorChanged = true;
                                }
                                else if (m_checkMeTests && std::find(m_checkMeTests->begin(), m_checkMeTests->end(), i + 1) != m_checkMeTests->end())
                                {
                                    ImGui::PushStyleColor(ImGuiCol_Text, UI_YELLOW);
                                    isColorChanged = true;
                                }

                                if (ImGui::Button(i == m_LastSelectedTest ? "*" : s, ImVec2(buttonWidth, 0.0f)) || isTestChanged)
                                {
                                    uint32_t test = isTestChanged ? m_LastSelectedTest : i;
                                    FILE* fp = fopen(path.c_str(), "rb");

                                    if (fp && fseek(fp, test * testByteSize, SEEK_SET) == 0)
                                    {
                                        size_t elemNum = fread(&m_Settings, sizeof(m_Settings), 1, fp);
                                        if (elemNum == 1)
                                            elemNum = fread(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);

                                        m_LastSelectedTest = test;

                                        // File read error
                                        if (elemNum != 1)
                                        {
                                            m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                                            m_Settings = m_DefaultSettings;
                                        }

                                        // Reset some settings to defaults to avoid a potential confusion
                                        m_Settings.debug = 0.0f;
                                        m_Settings.denoiser = REBLUR;
                                        m_Settings.DLSS = m_DefaultSettings.DLSS;
                                        m_ForceHistoryReset = true;
                                    }

                                    if (fp)
                                        fclose(fp);

                                    isTestChanged = false;
                                }

                                if (isColorChanged)
                                    ImGui::PopStyleColor();
                            }

                            if (i % 14 != 0)
                                ImGui::SameLine();

                            // "Add" button
                            if (ImGui::Button("Add"))
                            {
                                FILE* fp = fopen(path.c_str(), "ab");

                                if (fp)
                                {
                                    m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? -1.0 : 0.0;

                                    fwrite(&m_Settings, sizeof(m_Settings), 1, fp);
                                    fwrite(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);
                                    fclose(fp);

                                    m_TestNum = uint32_t(-1);
                                }
                            }

                            if ((i + 1) % 14 != 0)
                                ImGui::SameLine();

                            // "Del" button
                            snprintf(s, sizeof(s), "Del %u", m_LastSelectedTest + 1);
                            if (m_TestNum != uint32_t(-1) && m_LastSelectedTest != uint32_t(-1) && ImGui::Button(s))
                            {
                                std::vector<uint8_t> data;
                                utils::LoadFile(path, data);

                                FILE* fp = fopen(path.c_str(), "wb");

                                if (fp)
                                {
                                    for (i = 0; i < m_TestNum; i++)
                                    {
                                        if (i != m_LastSelectedTest)
                                            fwrite(&data[i * testByteSize], 1, testByteSize, fp);
                                    }

                                    fclose(fp);

                                    m_TestNum = uint32_t(-1);
                                }
                            }
                        }
                        ImGui::PopID();
                    }
                }
            }
            m_UiWidth = ImGui::GetWindowWidth();
        }
        ImGui::End();
    }

    // Update camera
    cBoxf cameraLimits = m_Scene.aabb;
    cameraLimits.Scale(2.0f);

    CameraDesc desc = {};
    desc.limits = cameraLimits;
    desc.aspectRatio = float(GetOutputResolution().x) / float(GetOutputResolution().y);
    desc.horizontalFov = RadToDeg(Atan(Tan(DegToRad(m_Settings.camFov) * 0.5f) * desc.aspectRatio * 9.0f / 16.0f) * 2.0f); // recalculate to ultra-wide if needed
    desc.nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    desc.farZ = 10000.0f * m_Settings.meterToUnitsMultiplier;
    desc.isCustomMatrixSet = m_Settings.animateCamera;
    desc.isLeftHanded = CAMERA_LEFT_HANDED;
    desc.orthoRange = m_Settings.ortho ? Tan(DegToRad(m_Settings.camFov) * 0.5f) * 3.0f * m_Settings.meterToUnitsMultiplier : 0.0f;
    GetCameraDescFromInputDevices(desc);

    const float animationSpeed = m_Settings.pauseAnimation ? 0.0f : (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
    const float scale = m_Settings.animatedObjectScale * m_Settings.meterToUnitsMultiplier / 2.0f;
    const float animationDelta = animationSpeed * m_Timer.GetFrameTime() * 0.001f;

    if (m_Settings.motionStartTime > 0.0)
    {
        float time = float(m_Timer.GetTimeStamp() - m_Settings.motionStartTime);
        float amplitude = 40.0f * m_Camera.state.motionScale;
        float period = 0.0003f * time * (m_Settings.emulateMotionSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.emulateMotionSpeed)) : (1.0f + m_Settings.emulateMotionSpeed));

        float3 localPos = m_Camera.state.mWorldToView.GetRow0().To3d();
        if (m_Settings.motionMode == 1)
            localPos = m_Camera.state.mWorldToView.GetRow1().To3d();
        else if (m_Settings.motionMode == 2)
            localPos = m_Camera.state.mWorldToView.GetRow2().To3d();
        else if (m_Settings.motionMode == 3)
        {
            float3 rows[3] = { m_Camera.state.mWorldToView.GetRow0().To3d(), m_Camera.state.mWorldToView.GetRow1().To3d(), m_Camera.state.mWorldToView.GetRow2().To3d() };
            float f = Sin(Pi(period * 3.0f));
            localPos = Normalize(f < 0.0f ? Lerp(rows[1], rows[0], float3(Abs(f))) : Lerp(rows[1], rows[2], float3(f)));
        }

        if (m_Settings.motionMode == 4)
        {
            float3 axisX = m_Camera.state.mWorldToView.GetRow0().To3d();
            float3 axisY = m_Camera.state.mWorldToView.GetRow1().To3d();
            float2 v = Rotate(float2(1.0f, 0.0f), Mod(Pi(period * 2.0f), Pi(2.0f)));
            localPos = (axisX * v.x + axisY * v.y) * amplitude / Pi(1.0f);
        }
        else
            localPos *= amplitude * (m_Settings.linearMotion ? WaveTriangle(period) - 0.5f : Sin(Pi(period)) * 0.5f);

        desc.dUser = localPos - m_PrevLocalPos;
        m_PrevLocalPos = localPos;
    }
    else if (m_Settings.motionStartTime == -1.0)
    {
        m_Settings.motionStartTime = m_Timer.GetTimeStamp();
        m_PrevLocalPos = float3::Zero();
    }

    m_Scene.Animate(animationSpeed, m_Timer.GetFrameTime(), m_Settings.animationProgress, m_Settings.activeAnimation, m_Settings.animateCamera ? &desc.customMatrix : nullptr);
    m_Camera.Update(desc, frameIndex);

    // Animate sun
    if (m_Settings.animateSun)
    {
        m_Settings.sunElevation += animationDelta * 10.0f;
        if (m_Settings.sunElevation > 180.0f)
            m_Settings.sunElevation -= 360.0f;
    }

    // Animate objects
    if (m_Settings.nineBrothers)
    {
        m_Settings.animatedObjectNum = 9;

        const float3& vRight = m_Camera.state.mViewToWorld.GetCol0().xmm;
        const float3& vTop = m_Camera.state.mViewToWorld.GetCol1().xmm;
        const float3& vForward = m_Camera.state.mViewToWorld.GetCol2().xmm;

        float3 basePos = ToFloat(m_Camera.state.globalPosition);

        for (int32_t i = -1; i <= 1; i++)
        {
            for (int32_t j = -1; j <= 1; j++)
            {
                const uint32_t index = (i + 1) * 3 + (j + 1);

                float x = float(i) * scale * 4.0f;
                float y = float(j) * scale * 4.0f;
                float z = 10.0f * scale * (CAMERA_LEFT_HANDED ? 1.0f : -1.0f);

                float3 pos = basePos + vRight * x + vTop * y + vForward * z;

                utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[index].instanceID];
                instance.position = ToDouble(pos);
                instance.rotation = m_Camera.state.mViewToWorld;
                instance.rotation.SetTranslation(float3::Zero());
                instance.rotation.AddScale(scale);
            }
        }
    }
    else if (m_Settings.animatedObjects)
    {
        for (int32_t i = 0; i < m_Settings.animatedObjectNum; i++)
        {
            float3 position;
            float4x4 transform = m_AnimatedInstances[i].Animate(animationDelta, scale, position);

            utils::Instance& instance = m_Scene.instances[m_AnimatedInstances[i].instanceID];
            instance.rotation = transform;
            instance.position = ToDouble(position);
        }
    }

    // Adjust settings if tracing mode has been changed to / from "probabilistic sampling"
    if (m_Settings.tracingMode != m_PrevSettings.tracingMode && (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC || m_PrevSettings.tracingMode == RESOLUTION_FULL_PROBABILISTIC))
    {
        nrd::ReblurSettings reblurDefaults = {};
        nrd::ReblurSettings relaxDefaults = {};

        if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
        {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        }
        else
        {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.diffusePrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.diffusePrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        }
    }

    // Print out information
    if (m_PrevSettings.resolutionScale != m_Settings.resolutionScale ||
        m_PrevSettings.tracingMode != m_Settings.tracingMode ||
        m_PrevSettings.rpp != m_Settings.rpp ||
        frameIndex == 0)
    {
        std::array<uint32_t, 4> rppScale = { 2, 1, 2, 2 };
        std::array<float, 4> wScale = { 1.0f, 1.0f, 0.5f, 0.5f };
        std::array<float, 4> hScale = { 1.0f, 1.0f, 1.0f, 0.5f };

        uint32_t pw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
        uint32_t ph = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
        uint32_t iw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale * wScale[m_Settings.tracingMode] + 0.5f);
        uint32_t ih = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale * hScale[m_Settings.tracingMode] + 0.5f);
        uint32_t rayNum = m_Settings.rpp * rppScale[m_Settings.tracingMode];
        float rpp = float(iw * ih * rayNum) / float(pw * ph);

        printf
        (
            "Output          : %ux%u\n"
            "Primary rays  : %ux%u\n"
            "Indirect rays : %ux%u x %u ray(s)\n"
            "Indirect rpp  : %.2f\n",
            GetOutputResolution().x, GetOutputResolution().y,
            pw, ph,
            iw, ih, rayNum,
            rpp
        );
    }

    if (m_PrevSettings.denoiser != m_Settings.denoiser || frameIndex == 0)
    {
        m_checkMeTests = nullptr;
        m_improveMeTests = nullptr;

        if (m_SceneFile.find("BistroInterior") != std::string::npos)
        {
            m_checkMeTests = &interior_checkMeTests;
            if (m_Settings.denoiser == REBLUR)
                m_improveMeTests = &REBLUR_interior_improveMeTests;
            else if (m_Settings.denoiser == RELAX)
                m_improveMeTests = &RELAX_interior_improveMeTests;
        }
    }
}

void Sample::LoadScene()
{
    std::string sceneFile = utils::GetFullPath("Cubes/Cubes.obj", utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, false));
    m_DefaultInstancesOffset = helper::GetCountOf(m_Scene.instances);

    sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, false));

    GenerateGeometry(m_Scene);

    m_ReblurSettings.antilagIntensitySettings.enable = true;

    if (m_SceneFile.find("BistroInterior") != std::string::npos)
    {
        m_Settings.exposure = 80.0f;
        m_Settings.emissionIntensity = 1.0f;
        m_Settings.emission = true;
        m_Settings.animatedObjectScale = 0.5f;
        m_Settings.sunElevation = 7.0f;
    }
    else if (m_SceneFile.find("BistroExterior") != std::string::npos)
    {
        m_Settings.exposure = 18.0f;
        m_Settings.emissionIntensity = 1.0f;
        m_Settings.emission = true;
    }
    else if (m_SceneFile.find("ShaderBalls") != std::string::npos)
        m_Settings.exposure = 1.7f;
    else if (m_SceneFile.find("ZeroDay") != std::string::npos)
    {
        m_Settings.exposure = 25.0f;
        m_Settings.emissionIntensity = 2.3f;
        m_Settings.emission = true;
        m_Settings.roughnessOverride = 0.07f;
        m_Settings.metalnessOverride = 0.25f;
        m_Settings.camFov = 75.0f;
        m_Settings.animationSpeed = -0.6f;
        m_Settings.sunElevation = -90.0f;
        m_Settings.sunAngularDiameter = 0.0f;
    }
}

void Sample::SetupAnimatedObjects()
{
    for (uint32_t i = 0; i < MAX_ANIMATED_INSTANCE_NUM; i++)
    {
        float3 position = Lerp(m_Scene.aabb.vMin, m_Scene.aabb.vMax, Rand::uf3(&m_FastRandState));

        AnimatedInstance animatedInstance = {};
        animatedInstance.instanceID = helper::GetCountOf(m_Scene.instances);
        animatedInstance.basePosition = position;
        animatedInstance.durationSec = Rand::uf1(&m_FastRandState) * 10.0f + 5.0f;
        animatedInstance.progressedSec = animatedInstance.durationSec * Rand::uf1(&m_FastRandState);
        animatedInstance.rotationAxis = Normalize(Rand::sf3(&m_FastRandState));
        animatedInstance.elipseAxis = Rand::sf3(&m_FastRandState) * 5.0f;
        animatedInstance.inverseDirection = Sign(Rand::sf1(&m_FastRandState));
        animatedInstance.inverseRotation = Sign(Rand::sf1(&m_FastRandState));
        m_AnimatedInstances.push_back(animatedInstance);

        uint32_t instanceIndex = i % m_DefaultInstancesOffset;
        const utils::Instance& instance = m_Scene.instances[instanceIndex];
        m_Scene.instances.push_back(instance);
    }
}

nri::Format Sample::CreateSwapChain()
{
    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.windowSystemType = GetWindowSystemType();
    swapChainDesc.window = GetWindow();
    swapChainDesc.commandQueue = m_CommandQueue;
    swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.verticalSyncInterval = m_VsyncInterval;
    swapChainDesc.width = (uint16_t)GetWindowResolution().x;
    swapChainDesc.height = (uint16_t)GetWindowResolution().y;
    swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

    nri::Format swapChainFormat = nri::Format::UNKNOWN;
    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum, swapChainFormat);

    nri::ClearValueDesc clearColor = {};
    nri::FrameBufferDesc frameBufferDesc = {};
    frameBufferDesc.colorAttachmentNum = 1;
    frameBufferDesc.colorClearValues = &clearColor;

    for (uint32_t i = 0; i < swapChainTextureNum; i++)
    {
        m_SwapChainBuffers.emplace_back();
        BackBuffer& backBuffer = m_SwapChainBuffers.back();

        backBuffer = {};
        backBuffer.texture = swapChainTextures[i];

        char name[32];
        snprintf(name, sizeof(name), "Texture::SwapChain#%u", i);
        NRI.SetTextureDebugName(*backBuffer.texture, name);

        nri::Texture2DViewDesc textureViewDesc = { backBuffer.texture, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat };
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, backBuffer.colorAttachment));

        frameBufferDesc.colorAttachments = &backBuffer.colorAttachment;
        NRI_ABORT_ON_FAILURE(NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, backBuffer.frameBufferUI));
    }

    return swapChainFormat;
}

void Sample::CreateCommandBuffers()
{
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, frame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer));
    }

    NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, m_OmmContext.commandAllocator));
    NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*m_OmmContext.commandAllocator, m_OmmContext.commandBuffer));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_OmmContext.fence));
}

void Sample::CreatePipelineLayoutAndDescriptorPool()
{
    const nri::DescriptorRangeDesc descriptorRanges0[] =
    {
        { 0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::ShaderStage::COMPUTE },
        { 0, 4, nri::DescriptorType::SAMPLER, nri::ShaderStage::COMPUTE },
    };

    const nri::DescriptorRangeDesc descriptorRanges1[] =
    {
        { 0, 10, nri::DescriptorType::TEXTURE, nri::ShaderStage::COMPUTE },
        { 0, 10, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::COMPUTE },
    };

    const uint32_t textureNum = helper::GetCountOf(m_Scene.materials) * TEXTURES_PER_MATERIAL;
    nri::DescriptorRangeDesc descriptorRanges2[] =
    {
        { 0, 2, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::ShaderStage::COMPUTE },
        { 2, 2, nri::DescriptorType::STRUCTURED_BUFFER, nri::ShaderStage::COMPUTE },
        { 4, textureNum, nri::DescriptorType::TEXTURE, nri::ShaderStage::COMPUTE, nri::VARIABLE_DESCRIPTOR_NUM, nri::DESCRIPTOR_ARRAY },
    };

    const nri::DescriptorSetDesc descriptorSetDesc[] =
    {
        { 0, descriptorRanges0, helper::GetCountOf(descriptorRanges0) },
        { 1, descriptorRanges1, helper::GetCountOf(descriptorRanges1), nullptr, 0, nri::DescriptorSetBindingBits::PARTIALLY_BOUND },
        { 2, descriptorRanges2, helper::GetCountOf(descriptorRanges2) },
    };

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
    pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
    pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

    nri::DescriptorPoolDesc descriptorPoolDesc = {};

    descriptorPoolDesc.descriptorSetMaxNum += BUFFERED_FRAME_MAX_NUM;
    descriptorPoolDesc.constantBufferMaxNum += descriptorRanges0[0].descriptorNum * BUFFERED_FRAME_MAX_NUM;
    descriptorPoolDesc.samplerMaxNum += descriptorRanges0[1].descriptorNum * BUFFERED_FRAME_MAX_NUM;

    descriptorPoolDesc.descriptorSetMaxNum += uint32_t(DescriptorSet::MAX_NUM);
    descriptorPoolDesc.textureMaxNum += descriptorRanges1[0].descriptorNum * uint32_t(DescriptorSet::MAX_NUM);
    descriptorPoolDesc.storageTextureMaxNum += descriptorRanges1[1].descriptorNum * uint32_t(DescriptorSet::MAX_NUM);

    descriptorPoolDesc.descriptorSetMaxNum += 1;
    descriptorPoolDesc.accelerationStructureMaxNum += descriptorRanges2[0].descriptorNum;
    descriptorPoolDesc.structuredBufferMaxNum += descriptorRanges2[1].descriptorNum;
    descriptorPoolDesc.textureMaxNum += descriptorRanges2[2].descriptorNum;

    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
}

void Sample::CreatePipelines()
{
    if (!m_Pipelines.empty())
    {
        NRI.WaitForIdle(*m_CommandQueue);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(*m_Pipelines[i]);
        m_Pipelines.clear();

        m_Reblur.CreatePipelines();
        m_Relax.CreatePipelines();
        m_Sigma.CreatePipelines();
        m_Reference.CreatePipelines();
    }

    utils::ShaderCodeStorage shaderCodeStorage;

    nri::ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = m_PipelineLayout;

    nri::Pipeline* pipeline = nullptr;
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    { // Pipeline::AmbientRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "AmbientRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PrimaryRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "PrimaryRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DirectLighting
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "DirectLighting.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::IndirectRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "IndirectRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Composition
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Composition.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Temporal
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Temporal.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Upsample
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Upsample.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::UpsampleNis
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "UpsampleNis.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PreDlss
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "PreDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::AfterDlss
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "AfterDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }
}

void Sample::CreateBottomLevelAccelerationStructures()
{
    for (const utils::Mesh& mesh : m_Scene.meshes)
    {
        const uint64_t vertexDataSize = mesh.vertexNum * sizeof(utils::Vertex);
        const uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);

        nri::Buffer* tempBuffer = nullptr;
        nri::Memory* tempMemory = nullptr;
        CreateUploadBuffer(vertexDataSize + indexDataSize, tempBuffer, tempMemory);

        uint8_t* data = (uint8_t*)NRI.MapBuffer(*tempBuffer, 0, nri::WHOLE_SIZE);
        memcpy(data, &m_Scene.vertices[mesh.vertexOffset], (size_t)vertexDataSize);
        memcpy(data + vertexDataSize, &m_Scene.indices[mesh.indexOffset], (size_t)indexDataSize);
        NRI.UnmapBuffer(*tempBuffer);

        nri::GeometryObject geometryObject = {};
        geometryObject.type = nri::GeometryType::TRIANGLES;
        geometryObject.flags = nri::BottomLevelGeometryBits::NONE;
        geometryObject.triangles.vertexBuffer = tempBuffer;
        geometryObject.triangles.vertexOffset = 0;
        geometryObject.triangles.vertexNum = mesh.vertexNum;
        geometryObject.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
        geometryObject.triangles.vertexStride = sizeof(utils::Vertex);
        geometryObject.triangles.indexBuffer = tempBuffer;
        geometryObject.triangles.indexOffset = vertexDataSize;
        geometryObject.triangles.indexNum = mesh.indexNum;
        geometryObject.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

        nri::AccelerationStructureDesc blasDesc = {};
        blasDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
        blasDesc.flags = BLAS_BUILD_BITS;
        blasDesc.instanceOrGeometryObjectNum = 1;
        blasDesc.geometryObjects = &geometryObject;

        nri::AccelerationStructure* blas = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, blasDesc, blas));
        m_BLASs.push_back(blas);

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*blas, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, blas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        BuildBottomLevelAccelerationStructure(*blas, &geometryObject, 1);

        NRI.DestroyBuffer(*tempBuffer);
        NRI.FreeMemory(*tempMemory);
    }
}

void Sample::CreateTopLevelAccelerationStructures()
{
    nri::Descriptor* descriptor = nullptr;
    nri::Memory* memory = nullptr;

    nri::AccelerationStructureDesc tlasDesc = {};
    tlasDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
    tlasDesc.flags = TLAS_BUILD_BITS;
    tlasDesc.instanceOrGeometryObjectNum = helper::GetCountOf(m_Scene.instances);

    { // Descriptor::World_AccelerationStructure
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, tlasDesc, m_WorldTlas));

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*m_WorldTlas, memoryDesc);

        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, m_WorldTlas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        NRI.CreateAccelerationStructureDescriptor(*m_WorldTlas, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::Light_AccelerationStructure
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, tlasDesc, m_LightTlas));

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*m_LightTlas, memoryDesc);

        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, m_LightTlas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        NRI.CreateAccelerationStructureDescriptor(*m_LightTlas, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }
}

void Sample::CreateSamplers()
{
    nri::Descriptor* descriptor = nullptr;

    { // Descriptor::LinearMipmapLinear_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = { nri::AddressMode::REPEAT, nri::AddressMode::REPEAT };
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;
        samplerDesc.mip = nri::Filter::LINEAR;
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::LinearMipmapNearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = { nri::AddressMode::REPEAT, nri::AddressMode::REPEAT };
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;
        samplerDesc.mip = nri::Filter::NEAREST;
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::Linear_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = { nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE };
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::Nearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = { nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE };
        samplerDesc.minification = nri::Filter::NEAREST;
        samplerDesc.magnification = nri::Filter::NEAREST;

        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Descriptors.push_back(descriptor);
    }
}

inline nri::Format ConvertFormatToTextureStorageCompatible(nri::Format format)
{
    switch (format)
    {
    case nri::Format::D16_UNORM:                return nri::Format::R16_UNORM;
    case nri::Format::D24_UNORM_S8_UINT:        return nri::Format::R24_UNORM_X8;
    case nri::Format::D32_SFLOAT:               return nri::Format::R32_SFLOAT;
    case nri::Format::D32_SFLOAT_S8_UINT_X24:   return nri::Format::R32_SFLOAT_X8_X24;
    case nri::Format::RGBA8_SRGB:               return nri::Format::RGBA8_UNORM;
    case nri::Format::BGRA8_SRGB:               return nri::Format::BGRA8_UNORM;
    default:                                    return format;
    }
}

void Sample::CreateResources(nri::Format swapChainFormat)
{
    // TODO: DLSS doesn't support R16 UNORM/SNORM
#if( NRD_MODE == OCCLUSION )
    nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::R16_SFLOAT : nri::Format::R16_UNORM;
    nri::Format dlssDataFormat = nri::Format::R16_SFLOAT;
#elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
    nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::RGBA16_SFLOAT : nri::Format::RGBA16_SNORM;
    nri::Format dlssDataFormat = nri::Format::R16_SFLOAT;
#else
    nri::Format dataFormat = nri::Format::RGBA16_SFLOAT;
    nri::Format dlssDataFormat = nri::Format::R11_G11_B10_UFLOAT;
#endif

#if( NRD_NORMAL_ENCODING == 0 )
    nri::Format normalFormat = nri::Format::RGBA8_UNORM;
#elif( NRD_NORMAL_ENCODING == 1 )
    nri::Format normalFormat = nri::Format::RGBA8_SNORM;
#elif( NRD_NORMAL_ENCODING == 2 )
    nri::Format normalFormat = nri::Format::R10_G10_B10_A2_UNORM;
#elif( NRD_NORMAL_ENCODING == 3 )
    nri::Format normalFormat = nri::Format::RGBA16_UNORM;
#elif( NRD_NORMAL_ENCODING == 4 )
    nri::Format normalFormat = nri::Format::RGBA16_SNORM;
#endif

    const uint16_t w = (uint16_t)m_RenderResolution.x;
    const uint16_t h = (uint16_t)m_RenderResolution.y;
    const uint64_t instanceDataSize = m_Scene.instances.size() * sizeof(InstanceData);
    const uint64_t worldScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*m_WorldTlas);
    const uint64_t lightScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*m_LightTlas);

    std::vector<DescriptorDesc> descriptorDescs;

    // Buffers (HOST_UPLOAD)
    CreateBuffer(descriptorDescs, "Buffer::GlobalConstants", m_ConstantBufferSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::CONSTANT_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::InstanceDataStaging", instanceDataSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::NONE);
    CreateBuffer(descriptorDescs, "Buffer::WorldTlasDataStaging", m_Scene.instances.size() * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ);
    CreateBuffer(descriptorDescs, "Buffer::LightTlasDataStaging", m_Scene.instances.size() * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ);

    // Buffers (DEVICE)
    CreateBuffer(descriptorDescs, "Buffer::PrimitiveData", m_Scene.primitives.size(), sizeof(PrimitiveData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::InstanceData", instanceDataSize / sizeof(InstanceData), sizeof(InstanceData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::WorldScratch", worldScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::LightScratch", lightScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);

    // Textures (DEVICE)
    CreateTexture(descriptorDescs, "Texture::Ambient", nri::Format::RGBA16_SFLOAT, 2, 2, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Motion", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Normal_Roughness", normalFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::PrimaryMipAndCurvature", nri::Format::RG8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::BaseColor_Metalness", nri::Format::RGBA8_SRGB, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectLighting", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectEmission", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TransparentLighting", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Shadow", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_ShadowData", nri::Format::RG16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Shadow_Translucency", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Validation", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Composed_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::DlssOutput", dlssDataFormat, (uint16_t)GetOutputResolution().x, (uint16_t)GetOutputResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::Final", swapChainFormat, (uint16_t)GetWindowResolution().x, (uint16_t)GetWindowResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::COPY_SOURCE);

    CreateTexture(descriptorDescs, "Texture::ComposedDiff_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ComposedSpec_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::TaaHistory", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TaaHistoryPrev", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);

#if( NRD_MODE == SH )
    CreateTexture(descriptorDescs, "Texture::Unfiltered_DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
#endif

    CreateTexture(descriptorDescs, "Texture::NisData1", nri::Format::RGBA16_SFLOAT, kFilterSize / 4, kPhaseCount, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);
    CreateTexture(descriptorDescs, "Texture::NisData2", nri::Format::RGBA16_SFLOAT, kFilterSize / 4, kPhaseCount, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    for (const utils::Texture* textureData : m_Scene.textures)
        CreateTexture(descriptorDescs, "", textureData->GetFormat(), textureData->GetWidth(), textureData->GetHeight(), textureData->GetMipNum(), textureData->GetArraySize(), nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    constexpr uint32_t offset = uint32_t(Buffer::UploadHeapBufferNum);

    // Bind memory
    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    resourceGroupDesc.bufferNum = offset;
    resourceGroupDesc.buffers = m_Buffers.data();

    size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = helper::GetCountOf(m_Buffers) - offset;
    resourceGroupDesc.buffers = m_Buffers.data() + offset;
    resourceGroupDesc.textureNum = helper::GetCountOf(m_Textures);
    resourceGroupDesc.textures = m_Textures.data();

    baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    // Create descriptors
    nri::Descriptor* descriptor = nullptr;
    for (const DescriptorDesc& desc : descriptorDescs)
    {
        if (desc.textureUsage == nri::TextureUsageBits::NONE)
        {
            if (desc.bufferUsage == nri::BufferUsageBits::CONSTANT_BUFFER)
            {
                for (uint32_t i = 0; i < BUFFERED_FRAME_MAX_NUM; i++)
                {
                    nri::BufferViewDesc bufferDesc = {};
                    bufferDesc.buffer = Get(Buffer::GlobalConstants);
                    bufferDesc.viewType = nri::BufferViewType::CONSTANT;
                    bufferDesc.offset = i * m_ConstantBufferSize;
                    bufferDesc.size = m_ConstantBufferSize;

                    NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferDesc, m_Frames[i].globalConstantBufferDescriptor));
                    m_Frames[i].globalConstantBufferOffset = bufferDesc.offset;
                }
            }
            else if (desc.bufferUsage & nri::BufferUsageBits::SHADER_RESOURCE)
            {
                const nri::BufferViewDesc viewDesc = { (nri::Buffer*)desc.resource, nri::BufferViewType::SHADER_RESOURCE, desc.format };
                NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, descriptor));
                m_Descriptors.push_back(descriptor);
            }

            NRI.SetBufferDebugName(*(nri::Buffer*)desc.resource, desc.debugName);
        }
        else
        {
            nri::Texture2DViewDesc viewDesc = { (nri::Texture*)desc.resource, desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_2D, desc.format };
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
            m_Descriptors.push_back(descriptor);

            if (desc.textureUsage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE)
            {
                viewDesc.format = ConvertFormatToTextureStorageCompatible(desc.format);
                viewDesc.viewType = desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D;
                NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
                m_Descriptors.push_back(descriptor);
            }

            NRI.SetTextureDebugName(*(nri::Texture*)desc.resource, desc.debugName);
        }
    }
}

void Sample::CreateDescriptorSets()
{
    nri::DescriptorSet* descriptorSet = nullptr;

    // Global constant buffer & samplers
    const nri::Descriptor* samplers[] =
    {
        Get(Descriptor::LinearMipmapLinear_Sampler),
        Get(Descriptor::LinearMipmapNearest_Sampler),
        Get(Descriptor::Linear_Sampler),
        Get(Descriptor::Nearest_Sampler),
    };

    for (Frame& frame : m_Frames)
    {
        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { &frame.globalConstantBufferDescriptor, 1 },
            { samplers, helper::GetCountOf(samplers) },
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &frame.globalConstantBufferDescriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));

        NRI.UpdateDescriptorRanges(*frame.globalConstantBufferDescriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AmbientRays1
        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Ambient_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 1, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PrimaryRays1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::ScramblingRanking1spp)),
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::SobolSequence)),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::PrimaryMipAndCurvature_StorageTexture),
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::DirectEmission_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::Unfiltered_Shadow_Translucency_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DirectLighting1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DirectEmission_Texture),
            Get(Descriptor::Shadow_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::DirectLighting_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::IndirectRays1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::PrimaryMipAndCurvature_Texture),
            Get(Descriptor::ComposedDiff_ViewZ_Texture),
            Get(Descriptor::ComposedSpec_ViewZ_Texture),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::TransparentLighting_StorageTexture),
            Get(Descriptor::Unfiltered_Diff_StorageTexture),
            Get(Descriptor::Unfiltered_Spec_StorageTexture),
#if( NRD_MODE == SH )
            Get(Descriptor::Unfiltered_DiffSh_StorageTexture),
            Get(Descriptor::Unfiltered_SpecSh_StorageTexture),
#endif
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Composition1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::DirectLighting_Texture),
            Get(Descriptor::TransparentLighting_Texture),
            Get(Descriptor::Ambient_Texture),
            Get(Descriptor::Diff_Texture),
            Get(Descriptor::Spec_Texture),
#if( NRD_MODE == SH )
            Get(Descriptor::DiffSh_Texture),
            Get(Descriptor::SpecSh_Texture),
#endif
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Composed_ViewZ_StorageTexture),
            Get(Descriptor::ComposedDiff_ViewZ_StorageTexture),
            Get(Descriptor::ComposedSpec_ViewZ_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
            Get(Descriptor::TaaHistoryPrev_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistory_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
            Get(Descriptor::TaaHistory_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistoryPrev_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistory_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistoryPrev_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::UpsampleNis1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistory_Texture),
            Get(Descriptor::NisData1),
            Get(Descriptor::NisData2),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::UpsampleNis1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistoryPrev_Texture),
            Get(Descriptor::NisData1),
            Get(Descriptor::NisData2),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PreDlss1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::DlssInput_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AfterDlss1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DlssOutput_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::RayTracing2
        std::vector<nri::Descriptor*> textures(m_Scene.materials.size() * TEXTURES_PER_MATERIAL);
        for (size_t i = 0; i < m_Scene.materials.size(); i++)
        {
            const size_t index = i * TEXTURES_PER_MATERIAL;
            const utils::Material& material = m_Scene.materials[i];

            textures[index] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.diffuseMapIndex));
            textures[index + 1] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.specularMapIndex));
            textures[index + 2] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.normalMapIndex));
            textures[index + 3] = Get(Descriptor((uint32_t)Descriptor::MaterialTextures + material.emissiveMapIndex));
        }

        const nri::Descriptor* structuredBuffers[] =
        {
            Get(Descriptor::InstanceData_Buffer),
            Get(Descriptor::PrimitiveData_Buffer)
        };

        const nri::Descriptor* accelerationStructures[] =
        {
            Get(Descriptor::World_AccelerationStructure),
            Get(Descriptor::Light_AccelerationStructure)
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 2, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, helper::GetCountOf(textures)));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc rtDescriptorRangeUpdateDesc[] =
        {
            { accelerationStructures, helper::GetCountOf(accelerationStructures) },
            { structuredBuffers, helper::GetCountOf(structuredBuffers) },
            { textures.data(), helper::GetCountOf(textures) }
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(rtDescriptorRangeUpdateDesc), rtDescriptorRangeUpdateDesc);
    }
}

void Sample::CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state)
{
    const nri::TextureDesc textureDesc = nri::Texture2D(format, width, height, mipNum, arraySize, usage);

    nri::Texture* texture = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, texture));
    m_Textures.push_back(texture);

    if (state != nri::AccessBits::UNKNOWN)
    {
        nri::TextureTransitionBarrierDesc transition = nri::TextureTransitionFromUnknown(texture, state, state == nri::AccessBits::SHADER_RESOURCE ? nri::TextureLayout::SHADER_RESOURCE : nri::TextureLayout::GENERAL);
        m_TextureStates.push_back(transition);
        m_TextureFormats.push_back(format);
    }

    descriptorDescs.push_back({ debugName, texture, format, usage, nri::BufferUsageBits::NONE, arraySize > 1 });
}

void Sample::CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage, nri::Format format)
{
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = elements * stride;
    bufferDesc.structureStride = (format == nri::Format::UNKNOWN && stride != 1) ? stride : 0;
    bufferDesc.usageMask = usage;

    nri::Buffer* buffer = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
    m_Buffers.push_back(buffer);

    descriptorDescs.push_back({ debugName, buffer, format, nri::TextureUsageBits::NONE, usage });
}

void Sample::CreateUploadBuffer(uint64_t size, nri::Buffer*& buffer, nri::Memory*& memory)
{
    const nri::BufferDesc bufferDesc = { size, 0, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ };
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryInfo(*buffer, nri::MemoryLocation::HOST_UPLOAD, memoryDesc);

    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = { memory, buffer };
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory)
{
    const uint64_t scratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(accelerationStructure);

    const nri::BufferDesc bufferDesc = { scratchBufferSize, 0, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE };
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryInfo(*buffer, nri::MemoryLocation::DEVICE, memoryDesc);

    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = { memory, buffer };
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::UploadStaticData()
{
    std::vector<PrimitiveData> primitiveData(m_Scene.primitives.size());
    uint32_t n = 0;
    for (const utils::Mesh& mesh : m_Scene.meshes)
    {
        uint32_t triangleNum = mesh.indexNum / 3;
        for (uint32_t j = 0; j < triangleNum; j++)
        {
            uint32_t primitiveIndex = mesh.indexOffset / 3 + j;
            const utils::Primitive& primitive = m_Scene.primitives[primitiveIndex];

            const utils::UnpackedVertex& v0 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3]];
            const utils::UnpackedVertex& v1 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3 + 1]];
            const utils::UnpackedVertex& v2 = m_Scene.unpackedVertices[mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3 + 2]];

            float2 n0 = Packed::EncodeUnitVector(float3(v0.normal), true);
            float2 n1 = Packed::EncodeUnitVector(float3(v1.normal), true);
            float2 n2 = Packed::EncodeUnitVector(float3(v2.normal), true);

            float2 t0 = Packed::EncodeUnitVector(float3(v0.tangent), true);
            float2 t1 = Packed::EncodeUnitVector(float3(v1.tangent), true);
            float2 t2 = Packed::EncodeUnitVector(float3(v2.tangent), true);

            PrimitiveData& data = primitiveData[n++];
            data.uv0 = Packed::sf2_to_h2(v0.uv[0], v0.uv[1]);
            data.uv1 = Packed::sf2_to_h2(v1.uv[0], v1.uv[1]);
            data.uv2 = Packed::sf2_to_h2(v2.uv[0], v2.uv[1]);

            data.n0oct = Packed::sf2_to_h2(n0.x, n0.y);
            data.n1oct = Packed::sf2_to_h2(n1.x, n1.y);
            data.n2oct = Packed::sf2_to_h2(n2.x, n2.y);

            data.t0oct = Packed::sf2_to_h2(t0.x, t0.y);
            data.t1oct = Packed::sf2_to_h2(t1.x, t1.y);
            data.t2oct = Packed::sf2_to_h2(t2.x, t2.y);

            data.b0s_b1s = Packed::sf2_to_h2(v0.tangent[3], v1.tangent[3]);
            data.b2s_worldToUvUnits = Packed::sf2_to_h2(v2.tangent[3], primitive.worldToUvUnits);
            data.curvature = primitive.curvature;
        }
    }

    // Gather subresources for read-only textures
    std::vector<nri::TextureSubresourceUploadDesc> subresources;
    subresources.push_back({ coef_scale_fp16, 1, (kFilterSize / 4) * 8, (kFilterSize / 4) * kPhaseCount * 8 });
    subresources.push_back({ coef_usm_fp16, 1, (kFilterSize / 4) * 8, (kFilterSize / 4) * kPhaseCount * 8 });
    for (const utils::Texture* texture : m_Scene.textures)
    {
        for (uint32_t layer = 0; layer < texture->GetArraySize(); layer++)
        {
            for (uint32_t mip = 0; mip < texture->GetMipNum(); mip++)
            {
                nri::TextureSubresourceUploadDesc subresource;
                texture->GetSubresource(subresource, mip, layer);

                subresources.push_back(subresource);
            }
        }
    }

    // Gather upload data for read-only textures
    std::vector<nri::TextureUploadDesc> textureData;
    textureData.push_back({ &subresources[0], Get(Texture::NisData1), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, 1, 1 });
    textureData.push_back({ &subresources[1], Get(Texture::NisData2), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, 1, 1 });
    size_t subresourceOffset = 2;

    for (size_t i = 0; i < m_Scene.textures.size(); i++)
    {
        const utils::Texture* texture = m_Scene.textures[i];
        uint16_t mipNum = texture->GetMipNum();
        uint16_t arraySize = texture->GetArraySize();

        textureData.push_back({ &subresources[subresourceOffset], Get((Texture)((size_t)Texture::MaterialTextures + i)), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, mipNum, arraySize });

        subresourceOffset += size_t(arraySize) * size_t(mipNum);
    }

    // Append textures without data to initialize initial state
    for (const nri::TextureTransitionBarrierDesc& state : m_TextureStates)
    {
        nri::TextureUploadDesc desc = {};
        desc.nextAccess = state.nextAccess;
        desc.nextLayout = state.nextLayout;
        desc.texture = (nri::Texture*)state.texture;

        textureData.push_back(desc);
    }

    // Buffer data
    nri::BufferUploadDesc dataDescArray[] =
    {
        { primitiveData.data(), helper::GetByteSizeOf(primitiveData), Get(Buffer::PrimitiveData), 0, nri::AccessBits::SHADER_RESOURCE },
    };

    // Upload data and apply states
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_CommandQueue, textureData.data(), helper::GetCountOf(textureData), dataDescArray, helper::GetCountOf(dataDescArray)));
}

void Sample::BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum)
{
    nri::Buffer* scratchBuffer = nullptr;
    nri::Memory* scratchBufferMemory = nullptr;
    CreateScratchBuffer(accelerationStructure, scratchBuffer, scratchBufferMemory);

    nri::CommandAllocator* commandAllocator = nullptr;
    NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, commandAllocator);

    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    NRI.BeginCommandBuffer(*commandBuffer, nullptr, 0);
    {
        NRI.CmdBuildBottomLevelAccelerationStructure(*commandBuffer, objectNum, objects, BLAS_BUILD_BITS, accelerationStructure, *scratchBuffer, 0);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;
    NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);

    NRI.WaitForIdle(*m_CommandQueue);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);
    NRI.DestroyBuffer(*scratchBuffer);
    NRI.FreeMemory(*scratchBufferMemory);
}

void Sample::BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex)
{
    bool isAnimatedObjects = m_Settings.animatedObjects;
    if (m_Settings.blink)
    {
        double period = 0.0003 * m_Timer.GetTimeStamp() * (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
        isAnimatedObjects &= WaveTriangle(period) > 0.5;
    }

    const uint64_t tlasCount = m_Scene.instances.size() - m_DefaultInstancesOffset;
    const uint64_t tlasDataSize = tlasCount * sizeof(nri::GeometryObjectInstance);
    const uint64_t tlasDataOffset = tlasDataSize * bufferedFrameIndex;
    const uint64_t instanceDataSize = tlasCount * sizeof(InstanceData);
    const uint64_t instanceDataOffset = instanceDataSize * bufferedFrameIndex;
    const uint64_t instanceCount = m_Scene.instances.size() - (m_AnimatedInstances.size() - m_Settings.animatedObjectNum * isAnimatedObjects);
    const uint64_t staticInstanceCount = m_Scene.instances.size() - m_AnimatedInstances.size();

    auto instanceData = (InstanceData*)NRI.MapBuffer(*Get(Buffer::InstanceDataStaging), instanceDataOffset, instanceDataSize);
    auto worldTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::WorldTlasDataStaging), tlasDataOffset, tlasDataSize);
    auto lightTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::LightTlasDataStaging), tlasDataOffset, tlasDataSize);

    uint32_t worldInstanceNum = 0;
    uint32_t lightInstanceNum = 0;
    m_HasTransparentObjects = false;
    for (size_t i = m_DefaultInstancesOffset; i < instanceCount; i++)
    {
        utils::Instance& instance = m_Scene.instances[i];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        if (material.IsOff()) // TODO: not an elegant way to skip "bad objects" (alpha channel is set to 0)
            continue;

        if (m_ShowOnlyAlphaTestedGeometry && material.IsAlphaOpaque() == false)
            continue;

        nri::AccelerationStructure* blas = m_EnableOmm ? GetMaskedBlas(GetInstanceHash(instance.meshIndex, instance.materialIndex)) : nullptr;

        assert(worldInstanceNum <= INSTANCE_ID_MASK);

        float4x4 mObjectToWorld = instance.rotation;
        mObjectToWorld.AddTranslation(m_Camera.GetRelative(instance.position));

        float4x4 mObjectToWorldPrev = instance.rotationPrev;
        mObjectToWorldPrev.AddTranslation(m_Camera.GetRelative(instance.positionPrev));

        // Use fp64 to avoid imprecision problems on close up views (InvertOrtho can't be used due to scaling factors)
        double4x4 mWorldToObjectd = ToDouble(mObjectToWorld);
        mWorldToObjectd.Invert();
        float4x4 mWorldToObject = ToFloat(mWorldToObjectd);

        float4x4 mWorldToWorldPrev = mObjectToWorldPrev * mWorldToObject;
        mWorldToWorldPrev.Transpose3x4();

        instance.positionPrev = instance.position;
        instance.rotationPrev = instance.rotation;

        mObjectToWorld.Transpose3x4();

        uint32_t flags = 0;
        if (material.IsEmissive()) // TODO: importance sampling can be significantly accelerated if ALL emissives will be placed into a single BLAS, which will be the only one in a special TLAS!
            flags = m_Settings.emission ? FLAG_EMISSION : FLAG_OPAQUE_OR_ALPHA_OPAQUE;
        else if (m_Settings.emissiveObjects && i > staticInstanceCount && (i % 3 == 0))
            flags = m_Settings.emission ? FLAG_FORCED_EMISSION : FLAG_OPAQUE_OR_ALPHA_OPAQUE;
        else if (material.IsTransparent())
        {
            flags = FLAG_TRANSPARENT;
            m_HasTransparentObjects = true;
        }
        else
            flags = FLAG_OPAQUE_OR_ALPHA_OPAQUE;

        uint32_t basePrimitiveIndex = mesh.indexOffset / 3;
        uint32_t instanceIdAndFlags = worldInstanceNum | (flags << FLAG_FIRST_BIT);

        uint32_t packedMaterial = Packed::uf4_to_uint<7, 7, 7, 0>(material.avgBaseColor);
        packedMaterial |= Packed::uf4_to_uint<11, 10, 6, 5>(float4(0.0f, 0.0f, material.avgSpecularColor.y, material.avgSpecularColor.z));

        instanceData->basePrimitiveIndex = basePrimitiveIndex;
        instanceData->baseTextureIndex = instance.materialIndex * TEXTURES_PER_MATERIAL;
        instanceData->mWorldToWorldPrev0 = mWorldToWorldPrev.col0;
        instanceData->mWorldToWorldPrev1 = mWorldToWorldPrev.col1;
        instanceData->mWorldToWorldPrev2 = mWorldToWorldPrev.col2;
        instanceData++;

        nri::GeometryObjectInstance tlasInstance = {};
        memcpy(tlasInstance.transform, mObjectToWorld.a16, sizeof(tlasInstance.transform));
        tlasInstance.instanceId = instanceIdAndFlags;
        tlasInstance.mask = flags;
        tlasInstance.shaderBindingTableLocalOffset = 0;
        tlasInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE | (material.IsAlphaOpaque() ? nri::TopLevelInstanceBits::NONE : nri::TopLevelInstanceBits::FORCE_OPAQUE);
        tlasInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_BLASs[instance.meshIndex], 0);

        blas = blas == nullptr ? m_BLASs[instance.meshIndex] : blas;
        tlasInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*blas, 0);

        if (flags & (FLAG_EMISSION | FLAG_FORCED_EMISSION))
        {
            *lightTlasData++ = tlasInstance;
            lightInstanceNum++;
        }

        *worldTlasData++ = tlasInstance;
        worldInstanceNum++;
    }

    NRI.UnmapBuffer(*Get(Buffer::InstanceDataStaging));
    NRI.UnmapBuffer(*Get(Buffer::WorldTlasDataStaging));
    NRI.UnmapBuffer(*Get(Buffer::LightTlasDataStaging));

    const nri::BufferTransitionBarrierDesc transitions[] =
    {
        { Get(Buffer::InstanceData), nri::AccessBits::SHADER_RESOURCE,  nri::AccessBits::COPY_DESTINATION },
    };

    nri::TransitionBarrierDesc transitionBarriers = {};
    transitionBarriers.buffers = transitions;
    transitionBarriers.bufferNum = helper::GetCountOf(transitions);
    NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

    NRI.CmdCopyBuffer(commandBuffer, *Get(Buffer::InstanceData), 0, 0, *Get(Buffer::InstanceDataStaging), 0, instanceDataOffset, instanceDataSize);
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, worldInstanceNum, *Get(Buffer::WorldTlasDataStaging), tlasDataOffset, TLAS_BUILD_BITS, *m_WorldTlas, *Get(Buffer::WorldScratch), 0);
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, lightInstanceNum, *Get(Buffer::LightTlasDataStaging), tlasDataOffset, TLAS_BUILD_BITS, *m_LightTlas, *Get(Buffer::LightScratch), 0);
}

void Sample::UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor)
{
    // Ambient accumulation
    const float maxSeconds = 0.5f;
    float maxAccumFrameNum = maxSeconds * 1000.0f / m_Timer.GetSmoothedFrameTime();
    m_AmbientAccumFrameNum = (m_AmbientAccumFrameNum + 1.0f) * globalResetFactor;
    m_AmbientAccumFrameNum = Min(m_AmbientAccumFrameNum, maxAccumFrameNum);

    const float3& sunDirection = GetSunDirection();

    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectWprev = uint32_t(m_RenderResolution.x * m_PrevSettings.resolutionScale + 0.5f);
    uint32_t rectHprev = uint32_t(m_RenderResolution.y * m_PrevSettings.resolutionScale + 0.5f);

    float emissionIntensity = m_Settings.emissionIntensity * float(m_Settings.emission);
    float baseMipBias = ((m_Settings.TAA || m_Settings.DLSS) ? -1.0f : 0.0f) + log2f(m_Settings.resolutionScale);

    float2 renderSize = float2(float(m_RenderResolution.x), float(m_RenderResolution.y));
    float2 outputSize = float2(float(GetOutputResolution().x), float(GetOutputResolution().y));
    float2 windowSize = float2(float(GetWindowResolution().x), float(GetWindowResolution().y));
    float2 rectSize = float2(float(rectW), float(rectH));
    float2 rectSizePrev = float2(float(rectWprev), float(rectHprev));
    float2 jitter = (m_Settings.cameraJitter ? m_Camera.state.viewportJitter : 0.0f) / rectSize;

    float3 viewDir = float3(m_Camera.state.mViewToWorld.GetCol2().xmm) * (CAMERA_LEFT_HANDED ? -1.0f : 1.0f);

    nrd::HitDistanceParameters hitDistanceParameters = {};
    hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;

    float minProbability = 0.0f;
    if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
    {
        nrd::HitDistanceReconstructionMode mode = nrd::HitDistanceReconstructionMode::OFF;
        if (m_Settings.denoiser == REBLUR)
            mode = m_ReblurSettings.hitDistanceReconstructionMode;
        else if (m_Settings.denoiser == RELAX)
            mode = m_RelaxSettings.hitDistanceReconstructionMode;

        // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
        if (mode == nrd::HitDistanceReconstructionMode::AREA_3X3)
            minProbability = 1.0f / 4.0f;
        else if (mode == nrd::HitDistanceReconstructionMode::AREA_5X5)
            minProbability = 1.0f / 16.0f;
    }

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const uint64_t rangeOffset = m_Frames[bufferedFrameIndex].globalConstantBufferOffset;
    nri::Buffer* globalConstants = Get(Buffer::GlobalConstants);
    auto data = (GlobalConstantBufferData*)NRI.MapBuffer(*globalConstants, rangeOffset, sizeof(GlobalConstantBufferData));
    {
        data->gViewToWorld = m_Camera.state.mViewToWorld;
        data->gViewToClip = m_Camera.state.mViewToClip;
        data->gWorldToView = m_Camera.state.mWorldToView;
        data->gWorldToViewPrev = m_Camera.statePrev.mWorldToView;
        data->gWorldToClip = m_Camera.state.mWorldToClip;
        data->gWorldToClipPrev = m_Camera.statePrev.mWorldToClip;
        data->gHitDistParams = float4(hitDistanceParameters.A, hitDistanceParameters.B, hitDistanceParameters.C, hitDistanceParameters.D);
        data->gCameraFrustum = m_Camera.state.frustum;
        data->gSunDirection_gExposure = sunDirection;
        data->gSunDirection_gExposure.w = m_Settings.exposure;
        data->gCameraOrigin_gMipBias = m_Camera.state.position;
        data->gCameraOrigin_gMipBias.w = baseMipBias + log2f(renderSize.x / outputSize.x);
        data->gTrimmingParams_gEmissionIntensity = GetSpecularLobeTrimming();
        data->gTrimmingParams_gEmissionIntensity.w = emissionIntensity;
        data->gViewDirection_gIsOrtho = float4(viewDir.x, viewDir.y, viewDir.z, m_Camera.m_IsOrtho);
        data->gWindowSize = windowSize;
        data->gInvWindowSize = float2(1.0f, 1.0f) / windowSize;
        data->gOutputSize = outputSize;
        data->gInvOutputSize = float2(1.0f, 1.0f) / outputSize;
        data->gRenderSize = renderSize;
        data->gInvRenderSize = float2(1.0f, 1.0f) / renderSize;
        data->gRectSize = rectSize;
        data->gInvRectSize = float2(1.0f, 1.0f) / rectSize;
        data->gRectSizePrev = rectSizePrev;
        data->gJitter = jitter;
        data->gNearZ = (CAMERA_LEFT_HANDED ? 1.0f : -1.0f) * NEAR_Z * m_Settings.meterToUnitsMultiplier;
        data->gAmbientAccumSpeed = 1.0f / (1.0f + m_AmbientAccumFrameNum);
        data->gAmbient = m_Settings.ambient ? 1.0f : 0.0f;
        data->gSeparator = m_Settings.separator;
        data->gRoughnessOverride = m_Settings.roughnessOverride;
        data->gMetalnessOverride = m_Settings.metalnessOverride;
        data->gUnitToMetersMultiplier = 1.0f / m_Settings.meterToUnitsMultiplier;
        data->gIndirectDiffuse = m_Settings.indirectDiffuse ? 1.0f : 0.0f;
        data->gIndirectSpecular = m_Settings.indirectSpecular ? 1.0f : 0.0f;
        data->gTanSunAngularRadius = Tan(DegToRad(m_Settings.sunAngularDiameter * 0.5f));
        data->gTanPixelAngularRadius = Tan(0.5f * DegToRad(m_Settings.camFov) / outputSize.x);
        data->gDebug = m_Settings.debug;
        data->gTransparent = (m_HasTransparentObjects && NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION && (m_Settings.onScreen == 0 || m_Settings.onScreen == 2)) ? 1.0f : 0.0f;
        data->gReference = m_Settings.denoiser == REFERENCE ? 1.0f : 0.0f;
        data->gUsePrevFrame = m_Settings.usePrevFrame;
        data->gMinProbability = minProbability;
        data->gDenoiserType = (uint32_t)m_Settings.denoiser;
        data->gDisableShadowsAndEnableImportanceSampling = (sunDirection.z < 0.0f && m_Settings.importanceSampling) ? 1 : 0;
        data->gOnScreen = m_Settings.onScreen + ((NRD_MODE == OCCLUSION || NRD_MODE == DIRECTIONAL_OCCLUSION) ? 3 : 0); // preserve original mapping
        data->gFrameIndex = frameIndex;
        data->gForcedMaterial = m_Settings.forcedMaterial;
        data->gUseNormalMap = m_Settings.normalMap ? 1 : 0;
        data->gIsWorldSpaceMotionEnabled = m_Settings.mvType == MV_3D ? 1 : 0;
        data->gTracingMode = m_Settings.tracingMode;
        data->gSampleNum = m_Settings.rpp;
        data->gBounceNum = m_Settings.bounceNum;
        data->gTAA = (m_Settings.denoiser != REFERENCE && m_Settings.TAA) ? 1 : 0;
        data->gResolve                                      = m_Settings.denoiser == REFERENCE ? false : m_Resolve;
        data->gPSR = m_Settings.PSR && m_Settings.tracingMode != RESOLUTION_HALF;
        data->gValidation = m_DebugNRD && m_ShowValidationOverlay && m_Settings.denoiser != REFERENCE && m_Settings.separator != 1.0f;
        data->gHighlightAhs = (uint32_t)m_Settings.highLightAhs;
        data->gAhsDynamicMipSelection = (uint32_t)m_Settings.ahsDynamicMipSelection;

        // NIS
        NISConfig config = {};
        NVScalerUpdateConfig
        (
            config, m_Settings.sharpness + Lerp((1.0f - m_Settings.sharpness) * 0.25f, 0.0f, (m_Settings.resolutionScale - 0.5f) * 2.0f),
            0, 0, rectW, rectH, m_RenderResolution.x, m_RenderResolution.y,
            0, 0, GetWindowResolution().x, GetWindowResolution().y, GetWindowResolution().x, GetWindowResolution().y,
            NISHDRMode::None
        );

        data->gNisDetectRatio = config.kDetectRatio;
        data->gNisDetectThres = config.kDetectThres;
        data->gNisMinContrastRatio = config.kMinContrastRatio;
        data->gNisRatioNorm = config.kRatioNorm;
        data->gNisContrastBoost = config.kContrastBoost;
        data->gNisEps = config.kEps;
        data->gNisSharpStartY = config.kSharpStartY;
        data->gNisSharpScaleY = config.kSharpScaleY;
        data->gNisSharpStrengthMin = config.kSharpStrengthMin;
        data->gNisSharpStrengthScale = config.kSharpStrengthScale;
        data->gNisSharpLimitMin = config.kSharpLimitMin;
        data->gNisSharpLimitScale = config.kSharpLimitScale;
        data->gNisScaleX = config.kScaleX;
        data->gNisScaleY = config.kScaleY;
        data->gNisDstNormX = config.kDstNormX;
        data->gNisDstNormY = config.kDstNormY;
        data->gNisSrcNormX = config.kSrcNormX;
        data->gNisSrcNormY = config.kSrcNormY;
        data->gNisInputViewportOriginX = config.kInputViewportOriginX;
        data->gNisInputViewportOriginY = config.kInputViewportOriginY;
        data->gNisInputViewportWidth = config.kInputViewportWidth;
        data->gNisInputViewportHeight = config.kInputViewportHeight;
        data->gNisOutputViewportOriginX = config.kOutputViewportOriginX;
        data->gNisOutputViewportOriginY = config.kOutputViewportOriginY;
        data->gNisOutputViewportWidth = config.kOutputViewportWidth;
        data->gNisOutputViewportHeight = config.kOutputViewportHeight;
    }
    NRI.UnmapBuffer(*globalConstants);
}

uint32_t Sample::BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM>& transitions)
{
    uint32_t n = 0;

    for (uint32_t i = 0; i < stateNum; i++)
    {
        const TextureState& state = states[i];
        nri::TextureTransitionBarrierDesc& transition = GetState(state.texture);

        bool isStateChanged = transition.nextAccess != state.nextAccess || transition.nextLayout != state.nextLayout;
        bool isStorageBarrier = transition.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE && state.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE;
        if (isStateChanged || isStorageBarrier)
            transitions[n++] = nri::TextureTransitionFromState(transition, state.nextAccess, state.nextLayout);
    }

    return n;
}

void Sample::RestoreBindings(nri::CommandBuffer& commandBuffer, const Frame& frame)
{
    NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
    NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *frame.globalConstantBufferDescriptorSet, nullptr);
    NRI.CmdSetDescriptorSet(commandBuffer, 2, *Get(DescriptorSet::RayTracing2), nullptr);
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    m_Profiler.BeginFrame();
    std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM> optimizedTransitions = {};

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];
    const bool isEven = !(frameIndex & 0x1);
    nri::TransitionBarrierDesc transitionBarriers = {};
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM)
    {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
    NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    // Global history reset
    float sunCurr = Smoothstep(-0.9f, 0.05f, Sin(DegToRad(m_Settings.sunElevation)));
    float sunPrev = Smoothstep(-0.9f, 0.05f, Sin(DegToRad(m_PrevSettings.sunElevation)));
    float resetHistoryFactor = 1.0f - Smoothstep(0.0f, 0.2f, Abs(sunCurr - sunPrev));

    if (m_PrevSettings.denoiser != m_Settings.denoiser)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.denoiser == REFERENCE && m_PrevSettings.tracingMode != m_Settings.tracingMode)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.ortho != m_Settings.ortho)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.onScreen != m_Settings.onScreen)
        m_ForceHistoryReset = true;

    // Sizes
    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectGridW = (rectW + 15) / 16;
    uint32_t rectGridH = (rectH + 15) / 16;
    uint32_t windowGridW = (GetWindowResolution().x + 15) / 16;
    uint32_t windowGridH = (GetWindowResolution().y + 15) / 16;

    // NRD settings
    if (m_Settings.adaptiveAccumulation)
    {
        bool isFastHistoryEnabled = m_Settings.maxAccumulatedFrameNum > m_Settings.maxFastAccumulatedFrameNum;

        float fps = 1000.0f / m_Timer.GetSmoothedFrameTime();
        float maxAccumulatedFrameNum = Clamp(ACCUMULATION_TIME * fps, 5.0f, float(MAX_HISTORY_FRAME_NUM));
        float maxFastAccumulatedFrameNum = isFastHistoryEnabled ? (maxAccumulatedFrameNum / 5.0f) : float(MAX_HISTORY_FRAME_NUM);

        m_Settings.maxAccumulatedFrameNum = int32_t(maxAccumulatedFrameNum + 0.5f);
        m_Settings.maxFastAccumulatedFrameNum = int32_t(maxFastAccumulatedFrameNum + 0.5f);
    }

    uint32_t maxAccumulatedFrameNum = uint32_t(m_Settings.maxAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    uint32_t maxFastAccumulatedFrameNum = uint32_t(m_Settings.maxFastAccumulatedFrameNum * resetHistoryFactor + 0.5f);

    nrd::CommonSettings commonSettings = {};
    memcpy(commonSettings.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));
    memcpy(commonSettings.viewToClipMatrixPrev, &m_Camera.statePrev.mViewToClip, sizeof(m_Camera.statePrev.mViewToClip));
    memcpy(commonSettings.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
    memcpy(commonSettings.worldToViewMatrixPrev, &m_Camera.statePrev.mWorldToView, sizeof(m_Camera.statePrev.mWorldToView));
    commonSettings.motionVectorScale[0] = m_Settings.mvType == MV_3D ? 1.0f : 1.0f / float(rectW);
    commonSettings.motionVectorScale[1] = m_Settings.mvType == MV_3D ? 1.0f : 1.0f / float(rectH);
    commonSettings.motionVectorScale[2] = m_Settings.mvType != MV_2D ? 1.0f : 0.0f;
    commonSettings.cameraJitter[0] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.x : 0.0f;
    commonSettings.cameraJitter[1] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.y : 0.0f;
    commonSettings.resolutionScale[0] = m_Settings.resolutionScale;
    commonSettings.resolutionScale[1] = m_Settings.resolutionScale;
    commonSettings.denoisingRange = GetDenoisingRange();
    commonSettings.disocclusionThreshold = m_Settings.disocclusionThreshold * 0.01f;
    commonSettings.splitScreen = m_Settings.denoiser == REFERENCE ? 1.0f : m_Settings.separator;
    commonSettings.debug = m_Settings.debug;
    commonSettings.frameIndex = frameIndex;
    commonSettings.accumulationMode = m_ForceHistoryReset ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.isMotionVectorInWorldSpace = m_Settings.mvType == MV_3D;
    commonSettings.isBaseColorMetalnessAvailable = true;
    commonSettings.enableValidation = m_DebugNRD && m_ShowValidationOverlay;

    // NRD user pool
    NrdUserPool userPool = {};
    {
        // Common
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_MV, { &GetState(Texture::Mv), GetFormat(Texture::Mv) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_NORMAL_ROUGHNESS, { &GetState(Texture::Normal_Roughness), GetFormat(Texture::Normal_Roughness) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_VIEWZ, { &GetState(Texture::ViewZ), GetFormat(Texture::ViewZ) });

        // (Optional) Needed to allow IN_MV modification on the NRD side
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_BASECOLOR_METALNESS, { &GetState(Texture::BaseColor_Metalness), GetFormat(Texture::BaseColor_Metalness) });

        // (Optional) Validation
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_VALIDATION, { &GetState(Texture::Validation), GetFormat(Texture::Validation) });

        // Diffuse
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, { &GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, { &GetState(Texture::Diff), GetFormat(Texture::Diff) });

        // Diffuse occlusion
#if( NRD_MODE == OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_HITDIST, { &GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_HITDIST, { &GetState(Texture::Diff), GetFormat(Texture::Diff) });
#endif

        // Diffuse SH
#if( NRD_MODE == SH )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH0, { &GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH1, { &GetState(Texture::Unfiltered_DiffSh), GetFormat(Texture::Unfiltered_DiffSh) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH0, { &GetState(Texture::Diff), GetFormat(Texture::Diff) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH1, { &GetState(Texture::DiffSh), GetFormat(Texture::DiffSh) });
#endif

        // Diffuse directional occlusion
#if( NRD_MODE == DIRECTIONAL_OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_DIRECTION_HITDIST, { &GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_DIRECTION_HITDIST, { &GetState(Texture::Diff), GetFormat(Texture::Diff) });
#endif

        // Specular
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, { &GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, { &GetState(Texture::Spec), GetFormat(Texture::Spec) });

        // Specular occlusion
#if( NRD_MODE == OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_HITDIST, { &GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_HITDIST, { &GetState(Texture::Spec), GetFormat(Texture::Spec) });
#endif

        // Specular SH
#if( NRD_MODE == SH )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH0, { &GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH1, { &GetState(Texture::Unfiltered_SpecSh), GetFormat(Texture::Unfiltered_SpecSh) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH0, { &GetState(Texture::Spec), GetFormat(Texture::Spec) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH1, { &GetState(Texture::SpecSh), GetFormat(Texture::SpecSh) });
#endif

        // SIGMA
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SHADOWDATA, { &GetState(Texture::Unfiltered_ShadowData), GetFormat(Texture::Unfiltered_ShadowData) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SHADOW_TRANSLUCENCY, { &GetState(Texture::Unfiltered_Shadow_Translucency), GetFormat(Texture::Unfiltered_Shadow_Translucency) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, { &GetState(Texture::Shadow), GetFormat(Texture::Shadow) });

        // REFERENCE
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_RADIANCE, { &GetState(Texture::Composed_ViewZ), GetFormat(Texture::Composed_ViewZ) });
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_RADIANCE, { &GetState(Texture::Composed_ViewZ), GetFormat(Texture::Composed_ViewZ) });
    }

    UpdateConstantBuffer(frameIndex, m_ForceHistoryReset ? 0.0f : resetHistoryFactor);

    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool, 0);
    {
        ProfilerContext* context01 = m_Profiler.BeginContext(&commandBuffer);
        { // TLAS
            helper::Annotation annotation(NRI, commandBuffer, "TLAS");

            BuildTopLevelAccelerationStructure(commandBuffer, bufferedFrameIndex);
        }

        // All-in-one pipeline layout
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);

        { // Ambient rays
            helper::Annotation annotation(NRI, commandBuffer, "Ambient rays");
            static uint32_t eventID = m_Profiler.AllocateEvent("Ambient rays");
            uint32_t timesampID = m_Profiler.BeginTimestamp(context01, eventID);

            const nri::BufferTransitionBarrierDesc bufferTransitions[] =
            {
                {Get(Buffer::InstanceData), nri::AccessBits::COPY_DESTINATION,  nri::AccessBits::SHADER_RESOURCE},
            };

            const TextureState transitions[] =
            {
                // Output
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            transitionBarriers.buffers = bufferTransitions;
            transitionBarriers.bufferNum = helper::GetCountOf(bufferTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
            transitionBarriers.bufferNum = 0;

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AmbientRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 0, *frame.globalConstantBufferDescriptorSet, nullptr);
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::AmbientRays1), nullptr);
            NRI.CmdSetDescriptorSet(commandBuffer, 2, *Get(DescriptorSet::RayTracing2), nullptr);

            NRI.CmdDispatch(commandBuffer, 2, 2, 1);
            m_Profiler.EndTimestamp(context01, timesampID);
        }

        { // Primary rays
            helper::Annotation annotation(NRI, commandBuffer, "Primary rays");
            static uint32_t eventID = m_Profiler.AllocateEvent("Primary + Shadows");
            uint32_t timesampID = m_Profiler.BeginTimestamp(context01, eventID);

            const TextureState transitions[] =
            {
                // Input
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::PrimaryMipAndCurvature, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Shadow_Translucency, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PrimaryRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::PrimaryRays1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            m_Profiler.EndTimestamp(context01, timesampID);
        }

        { // Shadow denoising
            helper::Annotation annotation(NRI, commandBuffer, "Shadow denoising");

            nrd::SigmaSettings shadowSettings = {};

            m_Sigma.SetMethodSettings(nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, &shadowSettings);
            m_Sigma.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);

            RestoreBindings(commandBuffer, frame);
        }

        { // Direct lighting
            helper::Annotation annotation(NRI, commandBuffer, "Direct lighting");

            const TextureState transitions[] =
            {
                // Input
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Shadow, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DirectLighting));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::DirectLighting1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Indirect rays
            helper::Annotation annotation(NRI, commandBuffer, "Indirect rays");
            static uint32_t eventID = m_Profiler.AllocateEvent("Indirect rays");
            uint32_t timesampID = m_Profiler.BeginTimestamp(context01, eventID);

            const TextureState transitions[] =
            {
                // Input
                {Texture::PrimaryMipAndCurvature, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::ComposedDiff_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::TransparentLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Spec, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
#if( NRD_MODE == SH )
                {Texture::Unfiltered_DiffSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_SpecSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
#endif
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::IndirectRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::IndirectRays1), nullptr);

            uint32_t rectWmod = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
            uint32_t rectHmod = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
            uint32_t rectGridWmod = (rectWmod + 15) / 16;
            uint32_t rectGridHmod = (rectHmod + 15) / 16;

            NRI.CmdDispatch(commandBuffer, rectGridWmod, rectGridHmod, 1);
            m_Profiler.EndTimestamp(context01, timesampID);
        }

        { // Diffuse & specular indirect lighting denoising
            helper::Annotation annotation(NRI, commandBuffer, "Indirect lighting denoising");

            float radiusResolutionScale = 1.0f;
            if (m_Settings.adaptRadiusToResolution)
                radiusResolutionScale = float(m_Settings.resolutionScale * m_RenderResolution.y) / 1440.0f;

            if (m_Settings.denoiser == REBLUR || m_Settings.denoiser == REFERENCE)
            {
                nrd::HitDistanceParameters hitDistanceParameters = {};
                hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;
                m_ReblurSettings.hitDistanceParameters = hitDistanceParameters;

                float resolutionScale = float(m_RenderResolution.x * m_Settings.resolutionScale) / float(GetOutputResolution().x);
                float antilagMaxThresholdScale = 0.25f + 0.75f / (1.0f + (m_Settings.rpp - 1) * 0.25f);

                nrd::AntilagHitDistanceSettings defaultAntilagHitDistanceSettings = {};
                m_ReblurSettings.antilagHitDistanceSettings.thresholdMin = defaultAntilagHitDistanceSettings.thresholdMin / resolutionScale;
                m_ReblurSettings.antilagHitDistanceSettings.thresholdMax = defaultAntilagHitDistanceSettings.thresholdMax * antilagMaxThresholdScale;
                m_ReblurSettings.antilagHitDistanceSettings.sigmaScale = defaultAntilagHitDistanceSettings.sigmaScale / resolutionScale;

                nrd::AntilagIntensitySettings defaultAntilagIntensitySettings = {};
                m_ReblurSettings.antilagIntensitySettings.thresholdMin = defaultAntilagIntensitySettings.thresholdMin / resolutionScale;
                m_ReblurSettings.antilagIntensitySettings.thresholdMax = defaultAntilagIntensitySettings.thresholdMax * antilagMaxThresholdScale;
                m_ReblurSettings.antilagIntensitySettings.sigmaScale = defaultAntilagIntensitySettings.sigmaScale / resolutionScale;

                // IMPORTANT: needs to be manually tuned based on the input range
                m_ReblurSettings.antilagIntensitySettings.sensitivityToDarkness = 0.01f;

                m_ReblurSettings.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_ReblurSettings.maxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_ReblurSettings.checkerboardMode = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                m_ReblurSettings.enableMaterialTestForDiffuse = true;
                m_ReblurSettings.enableMaterialTestForSpecular = false;

                nrd::ReblurSettings settings = m_ReblurSettings;
                settings.blurRadius *= radiusResolutionScale;
                settings.diffusePrepassBlurRadius *= radiusResolutionScale;
                settings.specularPrepassBlurRadius *= radiusResolutionScale;
                settings.historyFixStrideBetweenSamples *= radiusResolutionScale;

            #if( NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION )
                // High quality SG resolve allows to use more relaxed normal weights
                if (m_Resolve)
                    settings.lobeAngleFraction *= 1.333f;
            #endif

#if( NRD_MODE == OCCLUSION )
#if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, &m_ReblurSettings);
#else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_OCCLUSION, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR_OCCLUSION, &m_ReblurSettings);
#endif
#elif( NRD_MODE == SH )
#if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR_SH, &settings);
#else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SH, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR_SH, &m_ReblurSettings);
#endif
#elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION, &settings);
#else
#if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR, &settings);
#else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR, &m_ReblurSettings);
#endif
#endif

                m_Reblur.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);
            }
            else if (m_Settings.denoiser == RELAX)
            {
                m_RelaxSettings.diffuseMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.specularMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.specularMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.checkerboardMode = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                m_RelaxSettings.enableMaterialTestForDiffuse = true;
                m_RelaxSettings.enableMaterialTestForSpecular = false;

                nrd::RelaxDiffuseSpecularSettings settings = m_RelaxSettings;
                settings.diffusePrepassBlurRadius *= radiusResolutionScale;
                settings.specularPrepassBlurRadius *= radiusResolutionScale;
                settings.historyFixStrideBetweenSamples *= radiusResolutionScale;

#if( NRD_COMBINED == 1 )
                    #if( NRD_MODE == SH )
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SPECULAR_SH, &m_RelaxSettings);
                    #else
                m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SPECULAR, &m_RelaxSettings);
                    #endif
#else
                nrd::RelaxDiffuseSettings diffuseSettings = {};
                diffuseSettings.prepassBlurRadius = settings.diffusePrepassBlurRadius;
                diffuseSettings.diffuseMaxAccumulatedFrameNum = settings.diffuseMaxAccumulatedFrameNum;
                diffuseSettings.diffuseMaxFastAccumulatedFrameNum = settings.diffuseMaxFastAccumulatedFrameNum;
                diffuseSettings.diffusePhiLuminance = settings.diffusePhiLuminance;
                diffuseSettings.diffuseLobeAngleFraction = settings.diffuseLobeAngleFraction;
                diffuseSettings.historyFixEdgeStoppingNormalPower = settings.historyFixEdgeStoppingNormalPower;
                diffuseSettings.historyFixStrideBetweenSamples = settings.historyFixStrideBetweenSamples;
                diffuseSettings.historyFixFrameNum = settings.historyFixFrameNum;
                diffuseSettings.historyClampingColorBoxSigmaScale = settings.historyClampingColorBoxSigmaScale;
                diffuseSettings.spatialVarianceEstimationHistoryThreshold = settings.spatialVarianceEstimationHistoryThreshold;
                diffuseSettings.atrousIterationNum = settings.atrousIterationNum;
                diffuseSettings.minLuminanceWeight = settings.diffuseMinLuminanceWeight;
                diffuseSettings.depthThreshold = settings.depthThreshold;
                diffuseSettings.checkerboardMode = settings.checkerboardMode;
                diffuseSettings.hitDistanceReconstructionMode = settings.hitDistanceReconstructionMode;
                diffuseSettings.enableAntiFirefly = settings.enableAntiFirefly;
                diffuseSettings.enableMaterialTest = settings.enableMaterialTestForDiffuse;

                nrd::RelaxSpecularSettings specularSettings = {};
                specularSettings.prepassBlurRadius = settings.specularPrepassBlurRadius;
                specularSettings.specularMaxAccumulatedFrameNum = settings.specularMaxAccumulatedFrameNum;
                specularSettings.specularMaxFastAccumulatedFrameNum = settings.specularMaxFastAccumulatedFrameNum;
                specularSettings.specularPhiLuminance = settings.specularPhiLuminance;
                specularSettings.diffuseLobeAngleFraction = settings.diffuseLobeAngleFraction;
                specularSettings.specularLobeAngleFraction = settings.specularLobeAngleFraction;
                specularSettings.roughnessFraction = settings.roughnessFraction;
                specularSettings.specularVarianceBoost = settings.specularVarianceBoost;
                specularSettings.specularLobeAngleSlack = settings.specularLobeAngleSlack;
                specularSettings.historyFixEdgeStoppingNormalPower = settings.historyFixEdgeStoppingNormalPower;
                specularSettings.historyFixStrideBetweenSamples = settings.historyFixStrideBetweenSamples;
                specularSettings.historyFixFrameNum = settings.historyFixFrameNum;
                specularSettings.historyClampingColorBoxSigmaScale = settings.historyClampingColorBoxSigmaScale;
                specularSettings.spatialVarianceEstimationHistoryThreshold = settings.spatialVarianceEstimationHistoryThreshold;
                specularSettings.atrousIterationNum = settings.atrousIterationNum;
                specularSettings.minLuminanceWeight = settings.specularMinLuminanceWeight;
                specularSettings.depthThreshold = settings.depthThreshold;
                specularSettings.luminanceEdgeStoppingRelaxation = settings.luminanceEdgeStoppingRelaxation;
                specularSettings.normalEdgeStoppingRelaxation = settings.normalEdgeStoppingRelaxation;
                specularSettings.roughnessEdgeStoppingRelaxation = settings.roughnessEdgeStoppingRelaxation;
                specularSettings.checkerboardMode = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::BLACK : nrd::CheckerboardMode::OFF;
                specularSettings.hitDistanceReconstructionMode = settings.hitDistanceReconstructionMode;
                specularSettings.enableAntiFirefly = settings.enableAntiFirefly;
                specularSettings.enableRoughnessEdgeStopping = settings.enableRoughnessEdgeStopping;
                specularSettings.enableMaterialTest = settings.enableMaterialTestForSpecular;

                    #if( NRD_MODE == SH )
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SH, &diffuseSettings);
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_SPECULAR_SH, &specularSettings);
                    #else
                m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE, &diffuseSettings);
                m_Relax.SetMethodSettings(nrd::Method::RELAX_SPECULAR, &specularSettings);
#endif
                #endif

                m_Relax.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);
            }

            RestoreBindings(commandBuffer, frame);
        }

        { // Composition
            helper::Annotation annotation(NRI, commandBuffer, "Composition");

            const TextureState transitions[] =
            {
                // Input
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::TransparentLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Diff, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Spec, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
            #if( NRD_MODE == SH )
                {Texture::DiffSh, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::SpecSh, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
            #endif
                // Output
                {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ComposedDiff_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Composition));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::Composition1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        if (m_Settings.denoiser == REFERENCE)
        { // Reference
            helper::Annotation annotation(NRI, commandBuffer, "Reference denoising");

            commonSettings.resolutionScale[0] = 1.0f;
            commonSettings.resolutionScale[1] = 1.0f;
            commonSettings.splitScreen = m_Settings.separator;

            m_Reference.SetMethodSettings(nrd::Method::REFERENCE, &m_ReferenceSettings);
            m_Reference.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);

            RestoreBindings(commandBuffer, frame);
        }

        if (m_Settings.DLSS)
        {
            { // Pre
                helper::Annotation annotation(NRI, commandBuffer, "Pre Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::DlssInput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PreDlss));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::PreDlss1), nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // DLSS
                helper::Annotation annotation(NRI, commandBuffer, "Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::DlssInput, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                DlssDispatchDesc dlssDesc = {};
                dlssDesc.texOutput = { Get(Texture::DlssOutput), Get(Descriptor::DlssOutput_StorageTexture), GetFormat(Texture::DlssOutput), {GetOutputResolution().x, GetOutputResolution().y} };
                dlssDesc.texInput = { Get(Texture::DlssInput), Get(Descriptor::DlssInput_Texture), GetFormat(Texture::DlssInput), {m_RenderResolution.x, m_RenderResolution.y} };
                dlssDesc.texMv = { Get(Texture::Unfiltered_ShadowData), Get(Descriptor::Unfiltered_ShadowData_Texture), GetFormat(Texture::Unfiltered_ShadowData), {m_RenderResolution.x, m_RenderResolution.y} };
                dlssDesc.texDepth = { Get(Texture::ViewZ), Get(Descriptor::ViewZ_Texture), GetFormat(Texture::ViewZ), {m_RenderResolution.x, m_RenderResolution.y} };
                dlssDesc.sharpness = m_Settings.sharpness;
                dlssDesc.currentRenderResolution = { rectW, rectH };
                dlssDesc.motionVectorScale[0] = 1.0f;
                dlssDesc.motionVectorScale[1] = 1.0f;
                dlssDesc.jitter[0] = -m_Camera.state.viewportJitter.x;
                dlssDesc.jitter[1] = -m_Camera.state.viewportJitter.y;
                dlssDesc.reset = m_ForceHistoryReset;

                m_DLSS.Evaluate(&commandBuffer, dlssDesc);

                RestoreBindings(commandBuffer, frame); // TODO: is it needed?
            }

            { // After
                helper::Annotation annotation(NRI, commandBuffer, "After Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Validation, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AfterDlss));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::AfterDlss1), nullptr);

                NRI.CmdDispatch(commandBuffer, windowGridW, windowGridH, 1);
            }
        }
        else
        {
            const Texture taaSrc = isEven ? Texture::TaaHistoryPrev : Texture::TaaHistory;
            const Texture taaDst = isEven ? Texture::TaaHistory : Texture::TaaHistoryPrev;

            { // Temporal
                helper::Annotation annotation(NRI, commandBuffer, "Temporal");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {taaSrc, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {taaDst, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Temporal));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::Temporal1a : DescriptorSet::Temporal1b), nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // Upsample, copy and split screen
                helper::Annotation annotation(NRI, commandBuffer, "Upsample");

                const TextureState transitions[] =
                {
                    // Input
                    {taaDst, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Validation, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                bool isValidation = m_DebugNRD && m_ShowValidationOverlay;
                bool isNis = m_Settings.NIS && m_Settings.separator == 0.0f && !isValidation;
                if (isNis)
                {
                    NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::UpsampleNis));
                    NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::UpsampleNis1a : DescriptorSet::UpsampleNis1b), nullptr);

                    // See NIS_Config.h
                    windowGridW = (GetWindowResolution().x + 31) / 32;
                    windowGridH = (GetWindowResolution().y + 31) / 32;
                }
                else
                {
                    NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Upsample));
                    NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::Upsample1a : DescriptorSet::Upsample1b), nullptr);
                }

                NRI.CmdDispatch(commandBuffer, windowGridW, windowGridH, 1);
            }
        }

        const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
        const BackBuffer* backBuffer = &m_SwapChainBuffers[backBufferIndex];

        { // Copy to back-buffer
            const nri::TextureTransitionBarrierDesc copyTransitions[] =
            {
                nri::TextureTransitionFromState(GetState(Texture::Final), nri::AccessBits::COPY_SOURCE, nri::TextureLayout::GENERAL),
                nri::TextureTransition(backBuffer->texture, nri::AccessBits::UNKNOWN, nri::AccessBits::COPY_DESTINATION, nri::TextureLayout::UNKNOWN, nri::TextureLayout::GENERAL),
            };
            transitionBarriers.textures = copyTransitions;
            transitionBarriers.textureNum = helper::GetCountOf(copyTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdCopyTexture(commandBuffer, *backBuffer->texture, 0, nullptr, *Get(Texture::Final), 0, nullptr);
        }

        { // UI
            const nri::TextureTransitionBarrierDesc beforeTransitions = nri::TextureTransition(backBuffer->texture, nri::AccessBits::COPY_DESTINATION, nri::AccessBits::COLOR_ATTACHMENT, nri::TextureLayout::GENERAL, nri::TextureLayout::COLOR_ATTACHMENT);
            transitionBarriers.textures = &beforeTransitions;
            transitionBarriers.textureNum = 1;
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdBeginRenderPass(commandBuffer, *backBuffer->frameBufferUI, nri::RenderPassBeginFlag::SKIP_FRAME_BUFFER_CLEAR);
            RenderUserInterface(commandBuffer);
            NRI.CmdEndRenderPass(commandBuffer);

            const nri::TextureTransitionBarrierDesc afterTransitions = nri::TextureTransition(backBuffer->texture, nri::AccessBits::COLOR_ATTACHMENT, nri::AccessBits::UNKNOWN, nri::TextureLayout::COLOR_ATTACHMENT, nri::TextureLayout::PRESENT);
            transitionBarriers.textures = &afterTransitions;
            transitionBarriers.textureNum = 1;
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
        }
    }
    m_Profiler.EndFrame(&commandBuffer);
    NRI.EndCommandBuffer(commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &frame.commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;
    NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);

    NRI.SwapChainPresent(*m_SwapChain);

    NRI.QueueSignal(*m_CommandQueue, *m_FrameFence, 1 + frameIndex);

    // Cap FPS if requested
    float msLimit = m_Settings.limitFps ? 1000.0f / m_Settings.maxFps : 0.0f;
    double lastFrameTimeStamp = m_Timer.GetLastFrameTimeStamp();

    while (m_Timer.GetTimeStamp() - lastFrameTimeStamp < msLimit)
        ;
}