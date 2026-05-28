#version 450

layout(binding = 0) uniform sampler2D depthSampler;
layout(binding = 1) uniform sampler2D normalSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fresnel_out0;

const vec2 SizeScreen = vec2(1920.0, 864.0);
const float UI_SpecularIntensity = 1.0;
const float SkyDepth = 0.9999;
const float nearplan = 0.25;
const float farplan = 1000.0;
const float CFX_aspectRatio = SizeScreen.x / SizeScreen.y;
const float FOV = 45.0;
const vec2 pix = 1.0 / SizeScreen;

float read_depth(vec2 uv) {
    vec4 d = texture(depthSampler, uv);
    float z = float(uint(d.r * 255.0 + 0.5) +
                    (uint(d.g * 256.0 + 0.5) << 8u) +
                    (uint(d.b * 256.0 + 0.5) << 16u)) / 16777215.0;
    float linearDepth = (nearplan * farplan) / (farplan - z * (farplan - nearplan));
    return (linearDepth < farplan) ? (linearDepth / farplan) : 1.0;
}

// Reconstruct view-space position
vec3 UVtoPos(vec2 uv, float depth) {
    float Z = depth * farplan;
    vec3 pos;
    pos.z = Z;
    pos.xy = (uv * 2.0 - 1.0) * Z * tan(radians(FOV * 0.5));
    pos.x *= CFX_aspectRatio;
    return pos;
}

// Compute normal from depth gradient (no adaptive selection = real variation)
vec3 DepthNormal(vec2 uv) {
    float dc = read_depth(uv);
    float dr = read_depth(uv + vec2( pix.x, 0.0));
    float dl = read_depth(uv + vec2(-pix.x, 0.0));
    float du = read_depth(uv + vec2(0.0,  pix.y));
    float dd = read_depth(uv + vec2(0.0, -pix.y));

    // Pick closer neighbor to avoid crossing depth discontinuities
    float dx = (abs(dr - dc) < abs(dl - dc)) ? (dr - dc) : (dc - dl);
    float dy = (abs(du - dc) < abs(dd - dc)) ? (du - dc) : (dc - dd);

    // Screen-space gradient: dZ per pixel gives surface tilt
    // Scale to view-space: dx * farplan per uv-step
    float scaleX = 2.0 * farplan * tan(radians(FOV * 0.5)) * CFX_aspectRatio;
    float scaleY = 2.0 * farplan * tan(radians(FOV * 0.5));

    // Normal from depth gradient in view space (camera looks along +Z)
    vec3 n = normalize(vec3(-dx * scaleX, -dy * scaleY, 1.0));
    // Flip so it points toward camera (-Z in our view space)
    return vec3(n.xy, -n.z);
}

void main() {
    float depth = read_depth(fragTexCoord);

    if (depth >= SkyDepth) {
        fresnel_out0 = vec4(0.0);
        return;
    }

    // --- Use stored normal map ---
    vec3 normalTex = texture(normalSampler, fragTexCoord).rgb; // raw [0,1]
    vec3 normalA = normalize(normalTex * 2.0 - 1.0);          // decoded [-1,1]

    // [DEBUG] Output raw normalTex as color to identify coordinate space:
    //   Ground color tells us the coordinate system:
    //   Mostly green  (0.5, 1.0, 0.5) => world-space Y-up  normal=(0,1,0)
    //   Mostly blue   (0.5, 0.5, 1.0) => view-space  Z-fwd normal=(0,0,1)
    //   Mostly purple (0.5, 0.5, 1.0) and walls red/green => typical tangent space
    fresnel_out0 = vec4(normalTex, 1.0);
}

