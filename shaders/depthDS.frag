#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D albedoSampler;
layout(binding = 2) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outData;

const float nearplan = 0.25;
const float farplan = 1000.0;

void main() {
    // 1. Read and recompose packed 24-bit depth
    vec4 packedDepth = texture(depthSampler, fragTexCoord);
    float z = (packedDepth.r * 255.0 + packedDepth.g * 255.0 * 256.0 + packedDepth.b * 255.0 * 65536.0) / 16777215.0;
    
    // 2. Linearize depth
    float linearDepth = (nearplan * farplan) / (farplan - z * (farplan - nearplan));
    
    // 3. Normalize by farplane
    float depthNorm = (linearDepth < farplan) ? (linearDepth / farplan) : 1.0;
    
    // 4. Sample normal and albedo to get their W components (masks/roughness usually)
    vec4 normal = texture(normalSampler, fragTexCoord);
    vec4 albedo = texture(albedoSampler, fragTexCoord);
    
    // Output: R = LinearDepth, G = Normal.w, B = Albedo.w
    outData = vec4(depthNorm, normal.w, albedo.w, 1.0);
}
