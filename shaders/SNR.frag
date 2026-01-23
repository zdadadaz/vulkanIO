#version 450

layout(binding = 0) uniform sampler2D sRT_TNR_Out0;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = fragTexCoord;
    vec2 texelSize = 1.0 / textureSize(sRT_TNR_Out0, 0);

    // Simple 3x3 Gaussian Blur
    float kernel[9] = float[](
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0
    );

    vec3 blurColor = vec3(0.0);
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            blurColor += texture(sRT_TNR_Out0, uv + offset).rgb * kernel[(i+1)*3 + (j+1)];
        }
    }

    outColor = vec4(blurColor, 1.0);
}
