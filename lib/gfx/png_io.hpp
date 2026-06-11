#pragma once

#include <filesystem>
#include <optional>

#include "texture_convert.hpp"

namespace aurora::gfx::png {
std::optional<ConvertedTexture> load_png_file(const std::filesystem::path& path) noexcept;

// Writes 8-bit RGBA (or BGRA with bgra=true) pixel data to a PNG file.
// bytesPerRow may include padding beyond width * 4.
bool write_png_file(const std::filesystem::path& path, const uint8_t* data, uint32_t width, uint32_t height,
                    uint32_t bytesPerRow, bool bgra) noexcept;
}
