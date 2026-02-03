#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fresnel_out0;

// Constants approximating UI variables from CompleteRT
const vec2 SizeScreen = vec2(1920.0, 864.0);
const float UI_SpecularIntensity = 1.0;
const float SkyDepth = 0.9999;
const float nearplan = 0.25;
const float farplan = 1000.0;
const float CFX_aspectRatio = SizeScreen.x/SizeScreen.y;
const float FOV = 45.0;
const vec2 pix = 1.0 / SizeScreen;

vec3 UVtoPos(vec2 uv, float depth) {
    // Approximate view position reconstruction
    // Assuming simple perspective, FOV ~90 deg for now as we lack matrices
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth * farplan);
    ndc.xy *= ndc.z;
    ndc.x *= CFX_aspectRatio;
    ndc.xy *= tan(radians(FOV*0.5));
    // Simple projection:
    return ndc;
}

float read_depth(vec2 uv){
    vec4 packedDepth = texture(depthSampler, uv);
    float z = float(uint(packedDepth.r * 255.0 + 0.5) + 
                    (uint(packedDepth.g * 255.0 + 0.5) << 8u) + 
                    (uint(packedDepth.b * 255.0 + 0.5) << 16u)) / 16777215.0;
    
    // Linearize depth
    float linearDepth = (nearplan * farplan) / (farplan - z * (farplan - nearplan));
    
    // Normalize by farplane
    float depth = (linearDepth < farplan) ? (linearDepth / farplan) : 1.0;
    return depth;
}


vec3 ComputeNormal(vec2 uv) {

    vec2 dU = vec2(0, pix.y);
    vec2 dD = -vec2(0, pix.y);
    vec2 dL = vec2(pix.x, 0);
    vec2 dR = -vec2(pix.x, 0);
    vec3 u2, d2, l2, r2;

    const vec3 u = UVtoPos(uv + dD, read_depth(uv + dD)) ;
    const vec3 d = UVtoPos(uv + dU, read_depth(uv + dU)) ;
    const vec3 l = UVtoPos(uv + dL, read_depth(uv + dL)) ;
    const vec3 r = UVtoPos(uv + dR, read_depth(uv + dR)) ;

    u2 = UVtoPos(uv + dD + dD, read_depth(uv + dD + dD)) ;
    d2 = UVtoPos(uv + dU + dU, read_depth(uv + dU + dU)) ;
    l2 = UVtoPos(uv + dL + dL, read_depth(uv + dL + dL)) ;
    r2 = UVtoPos(uv + dR + dR, read_depth(uv + dR + dR)) ;

    u2 = u+ (u-u2);
    d2 = d+ (d-d2);
    l2 = l+ (l-l2);
    r2 = r+ (r-r2);

    const vec3 c = UVtoPos(uv, read_depth(uv));
    vec3 v = u-c;
    vec3 h = r-c;
    if(abs(d2.z-c.z) <abs(u2.z - c.z)) v = c-d;
    if(abs(l2.z-c.z) <abs(r2.z - c.z)) h = c-l;
    vec3 n = normalize(cross(v, h));
    return n;
}

void main() {
    float depth = read_depth(fragTexCoord);
    
    //vec3 normal = texture(normalSampler, fragTexCoord).rgb;
    //normal = normalize(normal * 2.0 - 1.0);
    vec3 normal = ComputeNormal(fragTexCoord);
    
    // Calculate View Direction (Eyedir)
    // Ideally: normalize(CameraPos - WorldPos).
    // In View Space: normalize(-ViewPos).
    // Here we approximate based on UV
    vec3 eyeDir = normalize(UVtoPos(fragTexCoord, depth)); // vector FROM eye TO pixel (assuming camera at origin)
    
    vec3 lightDir = reflect(eyeDir, normal);
    vec3 halfVec = normalize(lightDir + eyeDir);
    float dotLH = clamp(dot(lightDir, halfVec), 0.0, 1.0);
    
    // Calculate Fresnel
    float F0 = clamp(UI_SpecularIntensity * 0.1, 0.0, 1.0);
    float intensity = UI_SpecularIntensity * UI_SpecularIntensity + 1e-6;
    float fresnel = F0 + (1.0 - F0) * pow(dotLH, 5.0 / intensity);
    
    fresnel_out0 = vec4(clamp(fresnel, 0.0, 1.0));
}
