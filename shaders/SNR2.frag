#version 450

layout(binding = 0) uniform sampler2D inputSampler; // Output from SNR

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = fragTexCoord;
    vec2 texelSize = 1.0 / textureSize(inputSampler, 0);
    
    // Gaussian Kernel parameters
    // A 5x5 kernel requires a sigma around 1.0-2.0
    float sigma = 2.0;
    const int radius = 2; // 5x5 kernel centered at 0

    vec4 colorSum = vec4(0.0);
    float weightSum = 0.0;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            float weight = exp(-(float(x*x + y*y)) / (2.0 * sigma * sigma));
            
            colorSum += texture(inputSampler, uv + offset) * weight;
            weightSum += weight;
        }
    }

    outColor = colorSum / weightSum;
}
