#version 450

layout(binding = 0) uniform sampler2D texSampler;
layout(binding = 1) uniform sampler2D depthSampler; // This is now the output from depthDS.frag
layout(binding = 2) uniform sampler2D normalSampler; // Normals of the church scene

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const int MAX_STEPS = 100;
const float MAX_DIST = 100.0;
const float SURF_DIST = 0.001;
const float PI = 3.14159265359;

// RM Parameters (Must match depthDS.frag)
const float NEAR = 0.25;
const float FAR = 1000.0; 

float GetDist(vec3 p) {
    // Sphere at (0, 1, 6) with radius 1
    vec4 s = vec4(0, 1, 6, 1);
    float sphereDist = length(p - s.xyz) - s.w;
    
    // Plane at y=0 (Ground)
    float planeDist = p.y;
    
    // Combine
    return min(sphereDist, planeDist);
}

// Standard Raymarching for the SDF scene
float RayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO > MAX_DIST || dS < SURF_DIST) break;
    }
    return dO;
}

// Normal for the SDF scene
vec3 GetSDFNormal(vec3 p) {
    float d = GetDist(p);
    vec2 e = vec2(0.01, 0);
    vec3 n = d - vec3(
        GetDist(p - e.xyy),
        GetDist(p - e.yxy),
        GetDist(p - e.yyx));
    return normalize(n);
}

// Reconstruct View-Space Z from pre-processed depthDS buffer
float getSceneLinearDepth(vec2 uv) {
    // depthSampler now contains pre-linearized depth in the R channel (normalized [0,1] by FAR)
    float depthNorm = texture(depthSampler, uv).r;
    return depthNorm * FAR;
}

vec2 PostoUV(vec3 pos, vec3 ro) {
    vec3 p_rel = pos - ro;
    vec2 v_uv;
    v_uv.x = (p_rel.x / p_rel.z) / (1920.0 / 864.0);
    v_uv.y = -(p_rel.y / p_rel.z);
    return (v_uv + 1.0) * 0.5;
}

vec3 DoRayMarchSpecular(vec3 ro, vec3 position, vec3 raydir, float noiseZ) {
    const int RaySteps = 120;
    const int BackSteps = 15;
    const float RayInc = 1.0 + 1.0 / sqrt(float(RaySteps));
    const float rcpRaySteps = 1.0 / float(RaySteps);
    const float rcpRayInc = 1.0 / RayInc;
    
    float viewZ_start = position.z - ro.z;
    float bias = -viewZ_start * (1.0 / FAR) * 5.0;
    float ThicknessMul = -2.0; 
    
    float steplength = 0.04 * (1.0 + noiseZ * 0.5); 
    vec3 raypos = position;
    
    float j = 0.0;
    float hit_step = steplength;
    vec2 UVraypos;
    
    for(int i = 0; i < RaySteps; i++) {
        raypos += raydir * steplength;
        float currentViewZ = raypos.z - ro.z;
        if(currentViewZ < 0.0 || currentViewZ > FAR) break;

        UVraypos = PostoUV(raypos, ro);
        if(UVraypos.x < 0.0 || UVraypos.x > 1.0 || UVraypos.y < 0.0 || UVraypos.y > 1.0) break;
        
        float sceneDepth = getSceneLinearDepth(UVraypos);
        float error = sceneDepth - currentViewZ;
        
        if(error < bias && error > ThicknessMul * max(1.0, hit_step)) {
            if(j < float(BackSteps)) {
                raypos -= raydir * steplength;
                steplength *= rcpRaySteps;
                j++;
                i = 0; 
            } else {
                return texture(texSampler, UVraypos).rgb;
            }
        } else {
            steplength *= RayInc;
        }
        hit_step = steplength;
    }
    
    return vec3(-1.0);
}

float Hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec2 correctedUV = fragTexCoord;
    vec2 ndc = correctedUV * 2.0 - 1.0;
    float aspect = 1920.0 / 864.0;
    vec3 ro = vec3(0, 1, 0); 
    vec3 rd = normalize(vec3(ndc.x * aspect, -ndc.y, 1.0)); 
    
    vec4 sceneColor = texture(texSampler, fragTexCoord);
    float sceneZ = getSceneLinearDepth(fragTexCoord);
    
    float d = RayMarch(ro, rd);
    
    vec3 color;
    if(d < MAX_DIST && d < sceneZ) {
        vec3 p = ro + rd * d;
        vec3 n = GetSDFNormal(p);
        vec3 lightPos = vec3(2.0, 5.0, -1.0);
        vec3 l = normalize(lightPos - p);
        float dif = clamp(dot(n, l), 0.2, 1.0);
        vec3 baseColor = (p.y < 0.01) ? vec3(0.5) : vec3(0.4, 0.4, 0.6); 
        
        vec3 reflDir = reflect(rd, n);
        float noise = Hash(fragTexCoord);
        vec3 reflection = DoRayMarchSpecular(ro, p, reflDir, noise);
        
        if(reflection.r >= 0.0) {
            color = mix(baseColor * dif, reflection, 0.7);
        } else {
            color = baseColor * dif;
        }
    } else {
        color = sceneColor.rgb;
    }
    
    outColor = vec4(color, 1.0);
}
