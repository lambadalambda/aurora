#include "screenshot.hpp"

#include "png_io.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <cstdlib>
#include <filesystem>
#include <mutex>

static aurora::Module Log("aurora::gfx::screenshot");

namespace aurora::gfx::screenshot {

using webgpu::g_device;

namespace {
struct Pending {
  wgpu::Buffer buffer;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bytesPerRow = 0;
  bool bgra = false;
  std::filesystem::path path;
  bool copySubmitted = false;
  bool mapInFlight = false;
};

std::mutex g_mutex;
bool g_armed = false;
uint32_t g_captureIndex = 0;
Pending g_pending;

const char* trigger_path() {
  static const char* path = getenv("AURORA_SCREENSHOT_TRIGGER");
  return path;
}

std::filesystem::path output_dir() {
  static const std::filesystem::path dir = [] {
    const char* env = getenv("AURORA_SCREENSHOT_DIR");
    return std::filesystem::path(env != nullptr && env[0] != '\0' ? env : ".");
  }();
  return dir;
}
} // namespace

void check_trigger() {
  const char* trigger = trigger_path();
  if (trigger == nullptr) {
    return;
  }
  std::error_code ec;
  if (!std::filesystem::exists(trigger, ec)) {
    return;
  }
  std::filesystem::remove(trigger, ec);
  std::lock_guard lock{g_mutex};
  g_armed = true;
  Log.info("screenshot: armed");
}

void encode_frame(const wgpu::CommandEncoder& encoder, const webgpu::TextureWithSampler& src) {
  std::lock_guard lock{g_mutex};
  if (!g_armed) {
    return;
  }
  if (g_pending.copySubmitted || g_pending.mapInFlight) {
    Log.info("screenshot: capture already in flight (copySubmitted={} mapInFlight={})", g_pending.copySubmitted,
             g_pending.mapInFlight);
    return; // previous capture still in flight; stay armed and retry next frame
  }
  if (!src.texture || src.size.width == 0 || src.size.height == 0) {
    Log.info("screenshot: present source unavailable ({}x{})", src.size.width, src.size.height);
    return;
  }

  bool bgra = false;
  switch (src.texture.GetFormat()) {
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::RGBA8UnormSrgb:
    break;
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::BGRA8UnormSrgb:
    bgra = true;
    break;
  default:
    Log.error("unsupported present format {}", static_cast<uint32_t>(src.texture.GetFormat()));
    g_armed = false;
    return;
  }
  g_armed = false;

  // CopyTextureToBuffer requires bytesPerRow to be a multiple of 256.
  const uint32_t bytesPerRow = AURORA_ALIGN(src.size.width * 4, 256);
  const uint64_t byteSize = static_cast<uint64_t>(bytesPerRow) * src.size.height;
  const wgpu::BufferDescriptor descriptor{
      .label = "Screenshot readback buffer",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = byteSize,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  if (!buffer) {
    Log.error("failed to create readback buffer ({} bytes)", byteSize);
    return;
  }

  const wgpu::TexelCopyTextureInfo copySrc{
      .texture = src.texture,
  };
  const wgpu::TexelCopyBufferInfo copyDst{
      .layout =
          wgpu::TexelCopyBufferLayout{
              .bytesPerRow = bytesPerRow,
              .rowsPerImage = src.size.height,
          },
      .buffer = buffer,
  };
  const wgpu::Extent3D copySize{
      .width = src.size.width,
      .height = src.size.height,
      .depthOrArrayLayers = 1,
  };
  encoder.CopyTextureToBuffer(&copySrc, &copyDst, &copySize);

  g_pending = Pending{
      .buffer = std::move(buffer),
      .width = src.size.width,
      .height = src.size.height,
      .bytesPerRow = bytesPerRow,
      .bgra = bgra,
      .path = output_dir() / fmt::format("screenshot_{}.png", g_captureIndex++),
      .copySubmitted = true,
  };
}

void after_submit() {
  wgpu::Buffer buffer;
  uint64_t byteSize;
  {
    std::lock_guard lock{g_mutex};
    if (!g_pending.copySubmitted) {
      return;
    }
    g_pending.copySubmitted = false;
    g_pending.mapInFlight = true;
    buffer = g_pending.buffer;
    byteSize = static_cast<uint64_t>(g_pending.bytesPerRow) * g_pending.height;
  }
  buffer.MapAsync(wgpu::MapMode::Read, 0, byteSize, wgpu::CallbackMode::AllowSpontaneous,
                  [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                    std::lock_guard lock{g_mutex};
                    auto& p = g_pending;
                    if (status == wgpu::MapAsyncStatus::Success) {
                      const auto byteSize = static_cast<uint64_t>(p.bytesPerRow) * p.height;
                      const auto* data = static_cast<const uint8_t*>(p.buffer.GetConstMappedRange(0, byteSize));
                      if (data != nullptr && png::write_png_file(p.path, data, p.width, p.height, p.bytesPerRow, p.bgra)) {
                        Log.info("screenshot: wrote {} ({}x{})", p.path.string(), p.width, p.height);
                      } else {
                        Log.error("screenshot: failed to write {}", p.path.string());
                      }
                      p.buffer.Unmap();
                    } else {
                      Log.error("screenshot: map failed: {}", std::string_view{message});
                    }
                    p = {};
                  });
}

} // namespace aurora::gfx::screenshot
