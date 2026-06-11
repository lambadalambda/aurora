#pragma once

#include "common.hpp"

#include <webgpu/webgpu_cpp.h>

// Screen-space ambient occlusion: a mid-frame effect queued by the game
// between the 3D scene and 2D/UI drawing (gfx::queue_ssao). At encode time
// the pass that was current generates a half-resolution AO term from its
// depth attachment (depth-only, Crytek-style spiral taps) which is then
// multiplied onto the scene color.
namespace aurora::gfx::ssao {

struct Params {
  float invProj[16]; // inverse of the active GX projection (row-vector form)
  float width;       // render target size
  float height;
  float radius;    // world-space sample radius
  float intensity; // 0..1 occlusion strength
};

void initialize();
void shutdown();

// Builds parameters from the current GX projection and framebuffer size.
Params make_params(float radius, float intensity);

// Encodes AO generation + composite for a pass recorded with SSAO enabled.
struct RenderPassInfo; // see common.cpp RenderPass
void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView, const wgpu::TextureView& colorView,
            uint32_t msaaSamples, wgpu::Extent3D targetSize, const Range& uniformRange);

} // namespace aurora::gfx::ssao
