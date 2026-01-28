```glsl
#version 450

layout(binding = 0) uniform sampler2D sRT_RMOut;
layout(binding = 1) uniform sampler2D sRT_DepthDS;
layout(binding = 2) uniform sampler2D sRT_MV;
layout(binding = 3) uniform sampler2D sRT_TNRPrev0;
layout(binding = 4) uniform sampler2D sRT_TNRPrev1_prev;
layout(binding = 5) uniform sampler2D sRT_OriginalColor;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 TNR_out0;
layout(location = 1) out vec4 TNR_out1;

void main() {
    vec2 uv = fragTexCoord;
    
    // 1. Read MV and rearrange RGB888 into R10G10
    vec4 mv_raw = texture(sRT_MV, uv);
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
    
    // Process: mv0 = pow((mv_x-offset)*2,2); mv1 = pow((mv_y-offset)*2,2);
    float mv0 = pow((mv_x_norm - offset) * 2.0, 2.0);
    float mv1 = pow((mv_y_norm - offset) * 2.0, 2.0);
    
    // mv_out.x = (-mv0) if (mv_x < offset) ? mv0;
    vec2 motion;
    motion.x = (mv_x_norm < offset) ? -mv0 : mv0;
    motion.y = (mv_y_norm < offset) ? -mv1 : mv1;
    
    // 2. TAA Logic (Referenced from TemporalFilter in CompleteRT_Main.fxh)
    vec4 current = texture(sRT_RMOut, uv);
    vec4 depthDS = texture(sRT_DepthDS, uv);
    float depth = depthDS.r;
    
    vec2 pastUV = uv + motion;
    
    // Check if pastUV is inbound
    bool inbound = (pastUV.x >= 0.0 && pastUV.x <= 1.0 && pastUV.y >= 0.0 && pastUV.y <= 1.0);
    
    vec4 history = texture(sRT_TNRPrev0, pastUV);
    vec4 historyInfo = texture(sRT_TNRPrev1_prev, pastUV);
    
    float historyLength = historyInfo.x;
    float tm2 = historyInfo.y;
    float pastDepth = historyInfo.z;
    
    // Rejection logic (Simplified from TemporalFilter)
    // In original, they use normal and facing, but we simplify to depth for now
    float mask = 1.0;
    if (!inbound || abs(depth - pastDepth) > 0.01) {
        mask = 0.0;
    }
    
    historyLength *= mask;
    historyLength = min(historyLength, 32.0); // UI_MaxFrames equivalent
    
    historyLength += 1.0;
    
    // Update TM2 (second moment for variance/info)
    float currentLum = dot(current.rgb, vec3(0.299, 0.587, 0.114));
    if (historyLength > 4.0) {
        tm2 = mix(tm2, currentLum * currentLum, 1.0 / historyLength);
    } else {
        tm2 = currentLum * currentLum;
    }
    
    // Final TAA result
    vec3 result = mix(history.rgb, current.rgb, 1.0 / (historyLength + 1e-7));
    
    TNR_out0 = vec4(result, 1.0);
    TNR_out1 = vec4(historyLength, tm2, depth, 1.0);
}
