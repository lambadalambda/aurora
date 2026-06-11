#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
gfx::Range build_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges) noexcept;
u8 color_channel(GXChannelID id) noexcept;
// Uploads dirty matrix palette entries and refreshes g_gxState.mtxOffsets.
void flush_matrix_palette() noexcept;
// Texture dimensions + LOD bias as packed into per-draw uniforms.
Vec4<float> texture_size_bias(const gfx::TextureBind& tex);
}; // namespace aurora::gx
