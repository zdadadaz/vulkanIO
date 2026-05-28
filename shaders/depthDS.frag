#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D albedoSampler;
layout(binding = 2) uniform sampler2D normalSampler;
layout(binding = 3) uniform sampler2D newAlbedoSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outData;

const float nearplan = 0.25;
const float farplan = 1000.0;


float D24tofloat(vec4 d){
    return float(uint(d.r * 255.0 + 0.5) + (uint(d.g * 256.0 + 0.5) << 8u) + (uint(d.b * 256.0 + 0.5) << 16u)) / 16777215.0;
}

void main() {
    // 1. Read and recompose packed 24-bit depth
    vec4 packedDepth = textureLod(depthSampler, fragTexCoord, 0.0f);
    float z = D24tofloat(packedDepth);

    // 2. Linearize depth
    float linearDepth = (nearplan * farplan) / (farplan - z * (farplan - nearplan));
    
    // 3. Normalize by farplane
    float depthNorm = (linearDepth < farplan) ? (linearDepth / farplan) : 1.0;
    
    // 4. Sample normal and albedo to get their W components (masks/roughness usually)
    vec4 normal = texture(normalSampler, fragTexCoord);
    // vec4 albedo = texture(albedoSampler, fragTexCoord);
    vec4 newAlbedo = texture(newAlbedoSampler, fragTexCoord);
    
    // Output: R = LinearDepth, G = Normal.w, B = NewAlbedo.w
    outData = vec4(depthNorm, normal.w, newAlbedo.w, 1.0);
    // outData = vec4(z,z,z, 1.0);
}
