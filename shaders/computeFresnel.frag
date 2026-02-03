#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fresnel_out0;

// Constants approximating UI variables from CompleteRT
const float UI_SpecularIntensity = 1.0;
const float SkyDepth = 0.9999;
const float nearplan = 0.25;
const float farplan = 1000.0;

vec3 UVtoPos(vec2 uv, float depth) {
    // Approximate view position reconstruction
    // Assuming simple perspective, FOV ~90 deg for now as we lack matrices
    vec2 ndc = uv * 2.0 - 1.0;
    // float z = depth; // usage depends on depth buffer range (0..1)
    // Simple projection:
    return vec3(ndc, 1.0); // Simplified view direction vector essentially
}

void main() {
    // 1. Read and recompose packed 24-bit depth
    vec4 packedDepth = texture(depthSampler, fragTexCoord);
    float z = float(uint(packedDepth.r * 255.0 + 0.5) + 
                    (uint(packedDepth.g * 255.0 + 0.5) << 8u) + 
                    (uint(packedDepth.b * 255.0 + 0.5) << 16u)) / 16777215.0;
    
    // 2. Linearize depth
    float linearDepth = (nearplan * farplan) / (farplan - z * (farplan - nearplan));
    
    // 3. Normalize by farplane
    float depth = (linearDepth < farplan) ? (linearDepth / farplan) : 1.0;
    
    // Check discard condition (Sky/Far plane) with accurate depth
    if (depth > SkyDepth) {
        fresnel_out0 = vec4(0.0);
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
    
    vec3 lightDir = reflect(eyeDir, normal);
    vec3 halfVec = normalize(lightDir + eyeDir);
    float dotLH = clamp(dot(lightDir, halfVec), 0.0, 1.0);
    
    // Calculate Fresnel
    float F0 = clamp(UI_SpecularIntensity * 0.1, 0.0, 1.0);
    float intensity = UI_SpecularIntensity * UI_SpecularIntensity + 1e-6;
    float fresnel = F0 + (1.0 - F0) * pow(dotLH, 5.0 / intensity);
    
    fresnel_out0 = vec4(clamp(fresnel, 0.0, 1.0));
}
