#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out float fresnel_out0;

// Constants approximating UI variables from CompleteRT
const float UI_SpecularIntensity = 1.0;
const float SkyDepth = 0.9999;

vec3 UVtoPos(vec2 uv, float depth) {
    // Approximate view position reconstruction
    // Assuming simple perspective, FOV ~90 deg for now as we lack matrices
    vec2 ndc = uv * 2.0 - 1.0;
    // float z = depth; // usage depends on depth buffer range (0..1)
    // Simple projection:
    return vec3(ndc, 1.0); // Simplified view direction vector essentially
}

void main() {
    float depth = texture(depthSampler, fragTexCoord).r;
    
    // Check discard condition (Sky/Far plane)
    if (depth > SkyDepth) {
        fresnel_out0 = 0.0;
        return;
    }

    vec3 normal = texture(normalSampler, fragTexCoord).rgb;
    // Decode normals if they are [0,1]
    normal = normalize(normal * 2.0 - 1.0);
    
    // Calculate View Direction (Eyedir)
    // Ideally: normalize(CameraPos - WorldPos).
    // In View Space: normalize(-ViewPos).
    // Here we approximate based on UV
    vec2 ndc = fragTexCoord * 2.0 - 1.0;
    vec3 eyeDir = normalize(vec3(ndc, 1.0)); // vector FROM eye TO pixel (assuming camera at origin)
    // Wait, SpatialFilter4 says: Eyedir = normalize(UVtoPos(scaledUV)); where UVtoPos usually returns pos relative to camera.
    // So eyeDir is direction from Camera to Surface.
    
    // SpatialFilter4 Logic:
    // float3 LightDir = reflect(Eyedir, CNormals.xyz);
    // float3 HalfVec  = normalize(LightDir + Eyedir);
    // float  dotLH    = saturate(dot(LightDir, HalfVec));
    
    vec3 lightDir = reflect(eyeDir, normal);
    vec3 halfVec = normalize(lightDir + eyeDir);
    float dotLH = clamp(dot(lightDir, halfVec), 0.0, 1.0);
    
    // Calculate Fresnel
    float F0 = clamp(UI_SpecularIntensity * 0.1, 0.0, 1.0);
    float intensity = UI_SpecularIntensity * UI_SpecularIntensity + 1e-6;
    float fresnel = F0 + (1.0 - F0) * pow(dotLH, 5.0 / intensity);
    
    // Depth Fade (simplified)
    // float FadeFac = getDepthFade(depth);
    // fresnel = saturate(fresnel * FadeFac);
    
    fresnel_out0 = clamp(fresnel, 0.0, 1.0);
}
