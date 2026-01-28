#version 450

layout(binding = 0) uniform sampler2D sRT_TNR_Out0;
layout(binding = 1) uniform sampler2D sRT_Metadata; // DepthDS
layout(binding = 2) uniform sampler2D sRT_TNR_Info; // History info

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = fragTexCoord;
    vec2 texelSize = 1.0 / textureSize(sRT_TNR_Out0, 0);

    float centerDepth = texture(sRT_Metadata, uv).r;
    float historyLen = texture(sRT_TNR_Info, uv).r;

    // Adjust blur radius based on temporal stability (more history = stable = less blur needed)
    float radiusCheck = (historyLen > 10.0) ? 0.0 : 1.0; 

    // Simple Cross Bilateral Filter using Depth
    vec4 sum = vec4(0.0);
    float weightSum = 0.0;
    
    // 3x3 kernel
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            if (radiusCheck == 0.0 && (i!=0 || j!=0)) continue;

            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec2 sampleUV = uv + offset;
            
            vec4 sampleColor = texture(sRT_TNR_Out0, sampleUV);
            float sampleDepth = texture(sRT_Metadata, sampleUV).r;
            
            // Spatial weight (Gaussian)
            float spatial = exp(-(float(i*i + j*j)) / 2.0);
            
            // Range weight (Depth)
            float depthDiff = abs(centerDepth - sampleDepth);
            float range = exp(-(depthDiff * depthDiff) / 0.01); // Sensitivity
            
            float w = spatial * range;
            
            sum += sampleColor * w;
            weightSum += w;
        }
    }

    outColor = sum / weightSum;
}
