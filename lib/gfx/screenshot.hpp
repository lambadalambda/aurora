#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::webgpu {
struct TextureWithSampler;
} // namespace aurora::webgpu

// Trigger-file based screenshots of the present source, for headless
// debugging and automated verification:
//   AURORA_SCREENSHOT_TRIGGER=<path>  arm a screenshot whenever this file
//                                     exists (the file is deleted on pickup)
//   AURORA_SCREENSHOT_DIR=<dir>       output directory (default: cwd)
// Each capture writes <dir>/screenshot_<n>.png and logs the path.
namespace aurora::gfx::screenshot {

// Called once per frame; arms a capture if the trigger file exists.
void check_trigger();

// Encodes the GPU->CPU copy into the end-of-frame command encoder.
void encode_frame(const wgpu::CommandEncoder& encoder, const webgpu::TextureWithSampler& src);

// Starts the async map of the readback buffer after queue submission.
void after_submit();

} // namespace aurora::gfx::screenshot
