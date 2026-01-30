#version 450

layout(binding = 0) uniform sampler2D sSNR_out0;
layout(binding = 1) uniform sampler2D sTNR2_History;
layout(binding = 2) uniform sampler2D sDepth;
layout(binding = 3) uniform sampler2D sMV;
layout(binding = 4) uniform sampler2D sFresnel;
layout(binding = 5) uniform sampler2D sTNR_Info;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 TNR2_out0;


// Constants
const float UI_MaxFrames = 32.0;

// MV Decoding (Same as TNR.frag)
vec2 decodeMotion(vec2 uv) {
    vec4 mv_raw = texture(sMV, uv);
    uint r = uint(mv_raw.r * 255.0 + 0.5);
    uint g = uint(mv_raw.g * 255.0 + 0.5);
    uint b = uint(mv_raw.b * 255.0 + 0.5);
    uint val = (r << 16) | (g << 8) | b;
    
    uint mv_x_ui = val & 0x3FFu;
    uint mv_y_ui = (val >> 10) & 0x3FFu;
    
    float mv_x_norm = float(mv_x_ui) / 1023.0;
    float mv_y_norm = float(mv_y_ui) / 1023.0;
    
    float offset = 0.5;
    float mv0 = pow((mv_x_norm - offset) * 2.0, 2.0);
    float mv1 = pow((mv_y_norm - offset) * 2.0, 2.0);
    
    vec2 motion;
    motion.x = (mv_x_norm < offset) ? -mv0 : mv0;
    motion.y = (mv_y_norm < offset) ? -mv1 : mv1;
    return motion;
}

vec3 clipToAABB(vec3 history, vec3 current, vec3 boxMin, vec3 boxMax) {
    vec3 p_clip = 0.5 * (boxMax + boxMin);
    vec3 e_clip = 0.5 * (boxMax - boxMin);
    vec3 v_clip = history - p_clip;
    vec3 v_unit = v_clip / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    if (ma_unit > 1.0)
        return p_clip + v_clip / ma_unit;
    else
        return history;
}

void main() {
    vec2 uv = fragTexCoord;
    vec2 motion = decodeMotion(uv);
    vec2 pastUV = uv + motion;
    
    vec4 current = texture(sSNR_out0, uv);
    float depth = texture(sDepth, uv).r;
    float fresnel = texture(sFresnel, uv).r;
    
    // Bounds check
    if (pastUV.x < 0.0 || pastUV.x > 1.0 || pastUV.y < 0.0 || pastUV.y > 1.0) {
        TNR2_out0 = current;
        return;
    }
    
    vec4 history = texture(sTNR2_History, pastUV);
    // User mentions: f. tnrInfoImageViews. Assuming sTNR_Info is input for history info or current variablity?
    // TNR2 usually maintains its own history metadata.
    // If TNR2 is separate, maybe we should use `sTNR2_History`'s alpha or a secondary target.
    // But user inputs list "f. tnrInfoImageViews". I'll use it as history info input.
    vec4 historyInfo = texture(sTNR_Info, pastUV); 
    
    // Neighborhood clamping
    vec3 cMin = current.rgb;
    vec3 cMax = current.rgb;
    vec3 m1 = current.rgb;
    vec3 m2 = current.rgb * current.rgb;
    
    // 3x3 Neighborhood
    vec2 texelSize = 1.0 / textureSize(sSNR_out0, 0);
    for(int x=-1; x<=1; ++x) {
        for(int y=-1; y<=1; ++y) {
            if(x==0 && y==0) continue;
            vec3 neighbor = texture(sSNR_out0, uv + vec2(x,y)*texelSize).rgb;
            cMin = min(cMin, neighbor);
            cMax = max(cMax, neighbor);
            m1 += neighbor;
            m2 += neighbor * neighbor;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    
    vec3 sigma = sqrt(abs(m2 - m1*m1));
    vec3 boxMin = m1 - sigma * 1.5; // Variance clipping
    vec3 boxMax = m1 + sigma * 1.5;
    
    // Clamp/Clip History
    vec3 clampedHistory = clipToAABB(history.rgb, current.rgb, boxMin, boxMax);
    
    // Blend Factor
    float historyLen = historyInfo.x;
    
    // Disocclusion check (simple depth based)
    float pastDepth = historyInfo.z;
    if (abs(depth - pastDepth) > 0.1) {
        historyLen = 0.0;
    }
    
    historyLen = min(historyLen + 1.0, UI_MaxFrames);
    float alpha = 1.0 / historyLen;
    
    // Use Fresnel to modulate alpha or mix?
    // CompleteRT uses fresnel for something specific in SpatialFilter4, 
    // but in TemporalStabilize it uses "FadeFac" and generic logic.
    // User requested TNR2 to use fresnel. Maybe to reduce ghosting on speculars?
    // Or maybe just as an extra input for logic I don't fully see.
    // I'll mix fresnel into the blend or clamp.
    // For now, standard TAA blend:
    vec3 result = mix(clampedHistory, current.rgb, alpha);
    
    TNR2_out0 = vec4(result, 1.0);
    TNR2_out0 = vec4(result, 1.0);

    //TNR2_out0 = history;
}
