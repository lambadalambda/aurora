#pragma once

#include "../gfx/common.hpp"
#include "gx.hpp"

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  GXBindGroups bindGroups;
  uint32_t dstAlpha;
  // Ubershader fallback used while `pipeline` is still compiling; 0 when the
  // configuration is outside the interpreter's envelope or not needed.
  gfx::PipelineRef uberPipeline;
  gfx::Range uberUniformRange;
};

constexpr uint32_t GXPipelineConfigVersion = 15;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1;
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

wgpu::RenderPipeline create_pipeline([[maybe_unused]] const PipelineConfig& config);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
