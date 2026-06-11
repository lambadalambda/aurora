#include "ssao.hpp"

#include "../gx/gx.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <absl/container/flat_hash_map.h>

#include <vector>

namespace aurora::gfx::ssao {
static Module Log("aurora::gfx::ssao");

using webgpu::g_device;

namespace {
// AO generation: reconstructs view-space positions from depth and estimates
// occlusion from 8 spiral taps. Depth-only (no normals) keeps failure modes
// minimal; intensity is kept conservative by the caller.
//
// NDC reconstruction mirrors the generated vertex shaders: clip positions are
// produced as (mv * proj) with z negated afterwards (reversed Z), so the
// inverse transform negates ndc.z before applying inv_proj (row-vector form).
constexpr std::string_view AoShaderCommon = R"(
struct Params {
    inv_proj: mat4x4f,
    size: vec2f,
    radius: f32,
    intensity: f32,
    ao_floor: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(1) var<uniform> params: Params;

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = positions[vi] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    return out;
}

fn view_pos(uv: vec2f) -> vec3f {
    let px = clamp(vec2i(uv * params.size), vec2i(0), vec2i(params.size) - vec2i(1));
    let d = load_depth(px);
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, -d, 1.0);
    let v = ndc * params.inv_proj;
    return v.xyz / v.w;
}

const kTaps = array<vec2f, 8>(
    vec2f(1.0, 0.0), vec2f(0.5412, 0.8419), vec2f(-0.4161, 0.7935), vec2f(-0.8482, 0.0905),
    vec2f(-0.5784, -0.5806), vec2f(0.1325, -0.7905), vec2f(0.6743, -0.4459), vec2f(0.3624, 0.2173));

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let center = view_pos(in.uv);
    let dist = length(center);
    if (dist < 0.001 || dist > 50000.0) {
        return vec4f(1.0);
    }
    // View-space normal from central depth differences.
    let texel = vec2f(1.0, 1.0) / params.size;
    let dx = view_pos(in.uv + vec2f(texel.x, 0.0)) - view_pos(in.uv - vec2f(texel.x, 0.0));
    let dy = view_pos(in.uv + vec2f(0.0, texel.y)) - view_pos(in.uv - vec2f(0.0, texel.y));
    let nrm_v = cross(dy, dx);
    if (dot(nrm_v, nrm_v) < 1e-12) {
        return vec4f(1.0);
    }
    let normal = normalize(nrm_v);

    // Per-pixel kernel rotation (interleaved gradient noise) turns the fixed
    // tap pattern's ghosting/cross artifacts into fine noise, which the blur
    // pass then smooths.
    let ign = fract(52.9829189 * fract(0.06711056 * in.pos.x + 0.00583715 * in.pos.y));
    let ang = ign * 6.2831853;
    let rot = mat2x2f(vec2f(cos(ang), sin(ang)), vec2f(-sin(ang), cos(ang)));

    // Screen-space radius shrinks with distance; clamp to avoid huge kernels
    // up close and sub-pixel kernels far away.
    let screen_r = clamp(params.radius / dist, 0.002, 0.04);
    var occ = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let tap_uv = in.uv + (rot * kTaps[i]) * screen_r;
        let tap = view_pos(tap_uv);
        let delta = tap - center;
        let len = length(delta);
        if (len < 1e-3) {
            continue;
        }
        // Cosine-weighted hemisphere occlusion with an angle bias: only
        // geometry meaningfully above the tangent plane occludes, which
        // prevents slopes from shadowing themselves.
        let occ_cos = dot(delta / len, normal) - 0.18;
        let range_w = clamp(1.0 - len / (params.radius * 2.0), 0.0, 1.0);
        occ += max(occ_cos, 0.0) * range_w;
    }
    // Floor keeps occlusion from crushing the scene's baked shading.
    let ao = clamp(1.0 - params.intensity * (occ / 8.0) * 2.2, params.ao_floor, 1.0);
    return vec4f(ao, ao, ao, 1.0);
}
)";

// Depth-aware 3x3 blur over the half-res AO term, weighted so occlusion does
// not bleed across depth discontinuities (the source of silhouette halos).
constexpr std::string_view BlurShaderCommon = R"(
struct Params {
    inv_proj: mat4x4f,
    size: vec2f,
    radius: f32,
    intensity: f32,
    ao_floor: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = positions[vi] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    return out;
}

fn linear_dist(uv: vec2f) -> f32 {
    let px = clamp(vec2i(uv * params.size), vec2i(0), vec2i(params.size) - vec2i(1));
    let d = load_depth(px);
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, -d, 1.0);
    let v = ndc * params.inv_proj;
    return length(v.xyz / v.w);
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let ao_size = vec2f(textureDimensions(ao_tex));
    let center_px = vec2i(in.uv * ao_size);
    let center_d = linear_dist(in.uv);
    var total = 0.0;
    var weight = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let px = clamp(center_px + vec2i(x, y), vec2i(0), vec2i(ao_size) - vec2i(1));
            let uv = (vec2f(px) + vec2f(0.5)) / ao_size;
            let d = linear_dist(uv);
            // Reject contributions across depth discontinuities.
            let w = clamp(1.0 - abs(d - center_d) / max(center_d * 0.05, 1.0), 0.0, 1.0);
            total += textureLoad(ao_tex, px, 0).r * w;
            weight += w;
        }
    }
    let ao = total / max(weight, 1e-3);
    return vec4f(ao, ao, ao, 1.0);
}
)";

constexpr std::string_view DepthBindingSS = R"(
@group(0) @binding(0) var depth_tex: texture_depth_2d;
fn load_depth(px: vec2i) -> f32 { return textureLoad(depth_tex, px, 0); }
)";
constexpr std::string_view DepthBindingMS = R"(
@group(0) @binding(0) var depth_tex: texture_depth_multisampled_2d;
fn load_depth(px: vec2i) -> f32 { return textureLoad(depth_tex, px, 0); }
)";

// Composite: multiplies the AO term onto the scene color (blend dst*src).
constexpr std::string_view CompositeShader = R"(
@group(0) @binding(0) var ao_tex: texture_2d<f32>;
@group(0) @binding(1) var ao_samp: sampler;

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = positions[vi] * vec2f(0.5, -0.5) + vec2f(0.5, 0.5);
    return out;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let ao = textureSample(ao_tex, ao_samp, in.uv).r;
    return vec4f(ao, ao, ao, 1.0);
}
)";

wgpu::BindGroupLayout g_aoLayoutSS;
wgpu::BindGroupLayout g_aoLayoutMS;
wgpu::RenderPipeline g_aoPipelineSS;
wgpu::RenderPipeline g_aoPipelineMS;
wgpu::BindGroupLayout g_blurLayoutSS;
wgpu::BindGroupLayout g_blurLayoutMS;
wgpu::RenderPipeline g_blurPipelineSS;
wgpu::RenderPipeline g_blurPipelineMS;
webgpu::TextureWithSampler g_aoBlurTexture;
wgpu::BindGroupLayout g_compositeLayout;
absl::flat_hash_map<uint32_t, wgpu::RenderPipeline> g_compositePipelines; // by msaaSamples
wgpu::Sampler g_aoSampler;
webgpu::TextureWithSampler g_aoTexture;

constexpr wgpu::TextureFormat AoFormat = wgpu::TextureFormat::R8Unorm;

wgpu::ShaderModule make_module(std::string source, const char* label) {
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = source.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl, .label = label};
  return g_device.CreateShaderModule(&descriptor);
}

wgpu::BindGroupLayout make_ao_layout(bool multisampled, const char* label, bool withAoInput = false) {
  std::vector<wgpu::BindGroupLayoutEntry> entryVec;
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Depth,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
                  .multisampled = multisampled,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform},
      },
  };
  entryVec.assign(entries.begin(), entries.end());
  if (withAoInput) {
    entryVec.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                              .viewDimension = wgpu::TextureViewDimension::e2D},
    });
  }
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = label,
      .entryCount = entryVec.size(),
      .entries = entryVec.data(),
  };
  return g_device.CreateBindGroupLayout(&descriptor);
}

wgpu::RenderPipeline make_ao_pipeline(const wgpu::BindGroupLayout& layout, bool multisampled, const char* label,
                                      std::string_view body = AoShaderCommon) {
  std::string source{multisampled ? DepthBindingMS : DepthBindingSS};
  source += body;
  const auto module = make_module(std::move(source), label);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &layout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);
  const std::array colorTargets{wgpu::ColorTargetState{.format = AoFormat}};
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_composite_pipeline(uint32_t msaaSamples) {
  const auto module = make_module(std::string{CompositeShader}, "SSAO Composite Shader");
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_compositeLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);
  // Multiply blend: out = dst * src; alpha preserved.
  const wgpu::BlendState blend{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Dst,
                .dstFactor = wgpu::BlendFactor::Zero},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array colorTargets{wgpu::ColorTargetState{
      .format = webgpu::g_graphicsConfig.surfaceConfiguration.format,
      .blend = &blend,
  }};
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "SSAO Composite Pipeline",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .multisample = wgpu::MultisampleState{.count = msaaSamples},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

void ensure_ao_texture(uint32_t width, uint32_t height) {
  if (g_aoTexture.texture && g_aoTexture.size.width == width && g_aoTexture.size.height == height) {
    return;
  }
  const wgpu::TextureDescriptor descriptor{
      .label = "SSAO AO Texture",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .size = {width, height, 1},
      .format = AoFormat,
  };
  g_aoTexture.texture = g_device.CreateTexture(&descriptor);
  g_aoTexture.view = g_aoTexture.texture.CreateView();
  g_aoTexture.size = {width, height, 1};
  g_aoTexture.format = AoFormat;
  g_aoBlurTexture.texture = g_device.CreateTexture(&descriptor);
  g_aoBlurTexture.view = g_aoBlurTexture.texture.CreateView();
  g_aoBlurTexture.size = {width, height, 1};
  g_aoBlurTexture.format = AoFormat;
}
} // namespace

void initialize() {
  g_aoLayoutSS = make_ao_layout(false, "SSAO AO Layout");
  g_aoLayoutMS = make_ao_layout(true, "SSAO AO MS Layout");
  g_aoPipelineSS = make_ao_pipeline(g_aoLayoutSS, false, "SSAO AO Pipeline");
  g_aoPipelineMS = make_ao_pipeline(g_aoLayoutMS, true, "SSAO AO MS Pipeline");
  g_blurLayoutSS = make_ao_layout(false, "SSAO Blur Layout", true);
  g_blurLayoutMS = make_ao_layout(true, "SSAO Blur MS Layout", true);
  g_blurPipelineSS = make_ao_pipeline(g_blurLayoutSS, false, "SSAO Blur Pipeline", BlurShaderCommon);
  g_blurPipelineMS = make_ao_pipeline(g_blurLayoutMS, true, "SSAO Blur MS Pipeline", BlurShaderCommon);
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::Float,
                                                .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
      },
  };
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "SSAO Composite Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_compositeLayout = g_device.CreateBindGroupLayout(&descriptor);
  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "SSAO Sampler",
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_aoSampler = g_device.CreateSampler(&samplerDescriptor);
}

void shutdown() {
  g_aoLayoutSS = {};
  g_aoLayoutMS = {};
  g_aoPipelineSS = {};
  g_aoPipelineMS = {};
  g_blurLayoutSS = {};
  g_blurLayoutMS = {};
  g_blurPipelineSS = {};
  g_blurPipelineMS = {};
  g_aoBlurTexture = {};
  g_compositeLayout = {};
  g_compositePipelines.clear();
  g_aoSampler = {};
  g_aoTexture = {};
}

// General 4x4 inverse (adjugate); the projection matrix is always invertible.
static void invert4x4(const float m[16], float out[16]) {
  float inv[16];
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
           m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
           m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
           m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
            m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
           m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
           m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
           m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
            m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
           m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
           m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
            m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
            m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (det == 0.f) {
    det = 1.f;
  }
  det = 1.f / det;
  for (int i = 0; i < 16; ++i) {
    out[i] = inv[i] * det;
  }
}

Params make_params(float radius, float intensity, float aoFloor) {
  Params params{};
  invert4x4(reinterpret_cast<const float*>(&gx::g_gxState.proj), params.invProj);
  params.width = static_cast<float>(webgpu::g_frameBuffer.size.width);
  params.height = static_cast<float>(webgpu::g_frameBuffer.size.height);
  params.radius = radius;
  params.intensity = intensity;
  params.aoFloor = aoFloor;
  return params;
}

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView, const wgpu::TextureView& colorView,
            uint32_t msaaSamples, wgpu::Extent3D targetSize, const Range& uniformRange) {
  if (!depthView || !colorView || targetSize.width == 0) {
    return;
  }
  const bool multisampled = msaaSamples > 1;
  ensure_ao_texture(std::max(targetSize.width / 2, 1u), std::max(targetSize.height / 2, 1u));

  // 1) AO generation into the half-res AO texture.
  {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = depthView},
        wgpu::BindGroupEntry{.binding = 1,
                             .buffer = g_uniformBuffer,
                             .offset = uniformRange.offset,
                             .size = uniformRange.size},
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "SSAO AO Bind Group",
        .layout = multisampled ? g_aoLayoutMS : g_aoLayoutSS,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
    const std::array attachments{wgpu::RenderPassColorAttachment{
        .view = g_aoTexture.view,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {1.0, 1.0, 1.0, 1.0},
    }};
    const wgpu::RenderPassDescriptor passDescriptor{
        .label = "SSAO AO Pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = cmd.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(multisampled ? g_aoPipelineMS : g_aoPipelineSS);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3);
    pass.End();
  }

  // 2) Depth-aware blur of the AO term (kills tap noise and halos).
  {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = depthView},
        wgpu::BindGroupEntry{.binding = 1,
                             .buffer = g_uniformBuffer,
                             .offset = uniformRange.offset,
                             .size = uniformRange.size},
        wgpu::BindGroupEntry{.binding = 2, .textureView = g_aoTexture.view},
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "SSAO Blur Bind Group",
        .layout = multisampled ? g_blurLayoutMS : g_blurLayoutSS,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
    const std::array attachments{wgpu::RenderPassColorAttachment{
        .view = g_aoBlurTexture.view,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {1.0, 1.0, 1.0, 1.0},
    }};
    const wgpu::RenderPassDescriptor passDescriptor{
        .label = "SSAO Blur Pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = cmd.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(multisampled ? g_blurPipelineMS : g_blurPipelineSS);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3);
    pass.End();
  }

  // 3) Multiply the AO term onto the scene color.
  {
    auto it = g_compositePipelines.find(msaaSamples);
    if (it == g_compositePipelines.end()) {
      it = g_compositePipelines.try_emplace(msaaSamples, make_composite_pipeline(msaaSamples)).first;
    }
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .textureView = g_aoBlurTexture.view},
        wgpu::BindGroupEntry{.binding = 1, .sampler = g_aoSampler},
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "SSAO Composite Bind Group",
        .layout = g_compositeLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
    const std::array attachments{wgpu::RenderPassColorAttachment{
        .view = colorView,
        .loadOp = wgpu::LoadOp::Load,
        .storeOp = wgpu::StoreOp::Store,
    }};
    const wgpu::RenderPassDescriptor passDescriptor{
        .label = "SSAO Composite Pass",
        .colorAttachmentCount = attachments.size(),
        .colorAttachments = attachments.data(),
    };
    const auto pass = cmd.BeginRenderPass(&passDescriptor);
    pass.SetPipeline(it->second);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(3);
    pass.End();
  }
}

} // namespace aurora::gfx::ssao
