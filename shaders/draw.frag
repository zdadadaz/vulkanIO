#version 450

layout(binding = 0) uniform sampler2D finalSampler;
layout(binding = 1) uniform sampler2D colorSampler;
layout(binding = 2) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = fragTexCoord;
    
    // 1. Read MV and rearrange RGB888 into R10G10
    vec4 mv_raw = texture(finalSampler, uv);
    uint r = uint(mv_raw.r * 255.0 + 0.5);
    uint g = uint(mv_raw.g * 255.0 + 0.5);
    uint b = uint(mv_raw.b * 255.0 + 0.5);
    uint val = (r << 16) | (g << 8) | b;
    
    // Rearrange into R10G10 for mv_x and mv_y respectively
    uint mv_x_ui = val & 0x3FFu;         // bits 0-9
    uint mv_y_ui = (val >> 10) & 0x3FFu; // bits 10-19
    
    // Normalize into [0,1]
    float mv_x_norm = float(mv_x_ui) / 1023.0;
    float mv_y_norm = float(mv_y_ui) / 1023.0;
    
    float offset = 0.5;
    
    float diff_x = (mv_x_norm - offset) * 2.0;
    float diff_y = (mv_y_norm - offset) * 2.0;
    float mv0 = diff_x * diff_x;
    float mv1 = diff_y * diff_y;
    
    vec2 motion;
    motion.x = (mv_x_norm < offset) ? -mv0 : mv0;
    motion.y = (mv_y_norm < offset) ? -mv1 : mv1;
    
    // Visualize the real decoded motion vectors directly without over-scaling
    // This will show soft, dynamic colors representing the actual pixel movement!
    //outColor = vec4(abs(motion.x), abs(motion.y), 0.0, 1.0);
    outColor = vec4(fragTexCoord, 0, 1);
}
