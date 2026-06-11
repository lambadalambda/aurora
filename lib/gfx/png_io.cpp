#include "png_io.hpp"

#include <fstream>

#include "dds_io.hpp"
#include "png.h"
#include "../fs_helper.hpp"

static aurora::Module Log("aurora::gfx::png");

namespace aurora::gfx::png {

struct PngStructs {
  png_structp pStruct;
  png_infop pInfo;
  std::ifstream file;

  ~PngStructs() {
    png_destroy_read_struct(&pStruct, &pInfo, nullptr);
  }
};

static void readPngData(png_structp png, png_bytep data, const size_t length) {
  auto structs = static_cast<PngStructs*>(png_get_io_ptr(png));
  structs->file.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(length));

  if (structs->file.eof() || structs->file.fail()) {
    png_error(png, "file read failed!");
  }
}

std::optional<ConvertedTexture>
load_png_file(const std::filesystem::path& path) noexcept {
  PngStructs structs{};

  structs.file = std::move(std::ifstream(path, std::ifstream::in | std::ifstream::binary));
  if (!structs.file) {
    Log.error("failed to open file: {}", fs_path_to_string(path));
    return std::nullopt;
  }

  structs.pStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!structs.pStruct) {
    Log.error("png_create_read_struct failed");
    return std::nullopt;
  }

  structs.pInfo = png_create_info_struct(structs.pStruct);
  if (!structs.pInfo) {
    Log.error("png_create_info_struct failed");
    return std::nullopt;
  }

  // I'm scared of putting any locals after that setjmp.
  std::vector<png_bytep> rowPointers;
  ByteBuffer imageData{};
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type, compression_type, filter_type;
  size_t rowBytes;
  int i;

  if (setjmp(png_jmpbuf(structs.pStruct))) {
    Log.error("libpng encountered an error");
    return std::nullopt;
  }

  png_set_read_fn(structs.pStruct, &structs, readPngData);
  png_read_info(structs.pStruct, structs.pInfo);

  if (!png_get_IHDR(structs.pStruct, structs.pInfo, &width, &height, &bit_depth, &color_type, &interlace_type, &compression_type, &filter_type)) {
    Log.error("libpng unable to read IHDR");
    return std::nullopt;
  }

  // Always read as RGBA8.
  png_set_gray_to_rgb(structs.pStruct);
  png_set_filler(structs.pStruct, 0xFF, PNG_FILLER_AFTER);
  png_set_expand(structs.pStruct);
  png_set_strip_16(structs.pStruct);

  png_read_update_info(structs.pStruct, structs.pInfo);
  rowBytes = png_get_rowbytes(structs.pStruct, structs.pInfo);
  rowPointers.resize(height);

  imageData.append_zeroes(rowBytes * height);

  for (i = 0; i < height; i++) {
    rowPointers[i] = imageData.data() + i * rowBytes;
  }

  png_read_image(structs.pStruct, rowPointers.data());
  png_read_end(structs.pStruct, nullptr);

  return ConvertedTexture{
    .format = wgpu::TextureFormat::RGBA8Unorm,
    .width = width,
    .height = height,
    .mips = 1,
    .data = std::move(imageData)
  };
}

bool write_png_file(const std::filesystem::path& path, const uint8_t* data, uint32_t width, uint32_t height,
                    uint32_t bytesPerRow, bool bgra) noexcept {
  std::ofstream file(path, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
  if (!file) {
    Log.error("failed to open file for writing: {}", fs_path_to_string(path));
    return false;
  }

  png_structp pngWrite = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (pngWrite == nullptr) {
    return false;
  }
  png_infop pngInfo = png_create_info_struct(pngWrite);
  if (pngInfo == nullptr) {
    png_destroy_write_struct(&pngWrite, nullptr);
    return false;
  }
  if (setjmp(png_jmpbuf(pngWrite))) {
    png_destroy_write_struct(&pngWrite, &pngInfo);
    Log.error("libpng error while writing {}", fs_path_to_string(path));
    return false;
  }

  png_set_write_fn(
      pngWrite, &file,
      [](png_structp png, png_bytep out, const size_t length) {
        auto* stream = static_cast<std::ofstream*>(png_get_io_ptr(png));
        stream->write(reinterpret_cast<const char*>(out), static_cast<std::streamsize>(length));
        if (stream->fail()) {
          png_error(png, "file write failed!");
        }
      },
      [](png_structp png) {
        auto* stream = static_cast<std::ofstream*>(png_get_io_ptr(png));
        stream->flush();
      });

  // Write opaque RGB, dropping the input's alpha byte: EFB alpha is internal
  // blending data (often ~0) and makes screenshots render transparent.
  png_set_IHDR(pngWrite, pngInfo, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(pngWrite, pngInfo);
  png_set_filler(pngWrite, 0, PNG_FILLER_AFTER);
  if (bgra) {
    png_set_bgr(pngWrite);
  }

  std::vector<png_bytep> rowPointers(height);
  for (uint32_t i = 0; i < height; ++i) {
    rowPointers[i] = const_cast<png_bytep>(data + static_cast<size_t>(i) * bytesPerRow);
  }
  png_write_image(pngWrite, rowPointers.data());
  png_write_end(pngWrite, nullptr);
  png_destroy_write_struct(&pngWrite, &pngInfo);
  return true;
}
}