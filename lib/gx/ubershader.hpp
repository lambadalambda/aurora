#pragma once

#include "gx.hpp"
#include "pipeline.hpp"

// GX ubershader: a single WGSL shader that interprets the TEV/texgen/lighting
// configuration from uniform data at draw time. Used as a fallback while a
// draw's specialized pipeline is still compiling asynchronously, eliminating
// first-encounter pop-in for the covered configuration envelope.
//
// Uber pipelines are keyed only by render state (blend/depth/cull/MSAA) — the
// shader configuration travels in the uniform — so only a few dozen exist per
// game and they persist in the pipeline cache like any other pipeline.
namespace aurora::gx::uber {

constexpr uint32_t UberPipelineConfigVersion = 1;

// True if the interpreter covers this configuration. Conservative: draws
// outside the envelope simply keep today's skip-until-compiled behavior.
bool supports(const ShaderConfig& config) noexcept;

// Builds the fixed-layout interpreter uniform for the current GX state and
// the given shader configuration.
gfx::Range build_uniform(const ShaderConfig& config, u32 vtxStart, const BindGroupRanges& ranges) noexcept;

// Creates the uber pipeline for the given render state (shaderConfig is
// ignored and expected to be zeroed in the cache key).
wgpu::RenderPipeline create_pipeline(const PipelineConfig& config);

// Pipeline ref for the uber pipeline matching this render state. The
// caller must pass a config whose shaderConfig has been zeroed.
gfx::PipelineRef pipeline_ref(const PipelineConfig& config);

} // namespace aurora::gx::uber
