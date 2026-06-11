#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
bool sTraceChunksActive = false;
} // namespace detail

namespace {
FILE* sTraceFile = nullptr;
const char* sTraceTriggerPath = nullptr;
uint32_t sTraceSkipFrames = 0;
uint32_t sTraceFramesLeft = 0;
uint32_t sTraceFrameIndex = 0;
uint64_t sTraceFrameBytes = 0;

void trace_init_from_env() {
  const char* path = getenv("AURORA_GX_TRACE");
  if (path == nullptr || sTraceFile != nullptr) {
    return;
  }
  sTraceFile = fopen(path, "wb");
  if (sTraceFile == nullptr) {
    Log.error("gx-trace: failed to open '{}'", path);
    return;
  }
  const char* skip = getenv("AURORA_GX_TRACE_SKIP");
  const char* frames = getenv("AURORA_GX_TRACE_FRAMES");
  sTraceTriggerPath = getenv("AURORA_GX_TRACE_TRIGGER");
  sTraceSkipFrames = skip != nullptr ? static_cast<uint32_t>(strtoul(skip, nullptr, 10)) : 0;
  sTraceFramesLeft = frames != nullptr ? static_cast<uint32_t>(strtoul(frames, nullptr, 10)) : 120;
  fwrite("AURGXTR1", 1, 8, sTraceFile);
  detail::sTraceChunksActive = sTraceTriggerPath == nullptr && sTraceSkipFrames == 0;
  if (sTraceTriggerPath != nullptr) {
    Log.info("gx-trace: armed, will write {} frames to '{}' once '{}' exists", sTraceFramesLeft, path,
             sTraceTriggerPath);
  } else {
    Log.info("gx-trace: writing {} frames to '{}' after skipping {}", sTraceFramesLeft, path, sTraceSkipFrames);
  }
}
} // namespace

void trace_chunk(const uint8_t* data, uint32_t size, bool bigEndian) {
  if (size == 0) {
    return;
  }
  const uint8_t header[2]{0, static_cast<uint8_t>(bigEndian ? 1 : 0)};
  fwrite(header, 1, 2, sTraceFile);
  fwrite(&size, sizeof(size), 1, sTraceFile);
  fwrite(data, 1, size, sTraceFile);
  sTraceFrameBytes += size;
}

void trace_end_frame() {
  if (sTraceFile == nullptr) {
    return;
  }
  if (sTraceTriggerPath != nullptr) {
    FILE* trigger = fopen(sTraceTriggerPath, "rb");
    if (trigger == nullptr) {
      return; // still waiting for the trigger file
    }
    fclose(trigger);
    sTraceTriggerPath = nullptr;
    detail::sTraceChunksActive = sTraceSkipFrames == 0;
    Log.info("gx-trace: triggered, capturing {} frames", sTraceFramesLeft);
    return;
  }
  ++sTraceFrameIndex;
  if (sTraceSkipFrames > 0) {
    --sTraceSkipFrames;
    if (sTraceSkipFrames % 120 == 0) {
      Log.info("gx-trace: skipping, {} frames left before capture", sTraceSkipFrames);
    }
    detail::sTraceChunksActive = sTraceSkipFrames == 0;
    return;
  }
  const uint8_t endFrame = 1;
  fwrite(&endFrame, 1, 1, sTraceFile);
  if (sTraceFrameIndex % 30 == 0) {
    Log.info("gx-trace: frame {} captured ({} KiB this frame)", sTraceFrameIndex, sTraceFrameBytes / 1024);
  }
  sTraceFrameBytes = 0;
  if (--sTraceFramesLeft == 0) {
    fclose(sTraceFile);
    sTraceFile = nullptr;
    detail::sTraceChunksActive = false;
    Log.info("gx-trace: capture complete");
  }
}

void init() {
  constexpr uint32_t initialCapacity = 64 * 1024;
  free(detail::sBufferData);
  detail::sBufferData = static_cast<uint8_t*>(malloc(initialCapacity));
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  trace_init_from_env();
}

void write_data_grow(const void* data, uint32_t length) {
  uint32_t needed = detail::sBufferSize + length;
  uint32_t newCap = std::max(detail::sBufferCapacity * 2, needed);
  detail::sBufferData = static_cast<uint8_t*>(realloc(detail::sBufferData, newCap));
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCap;
}

void patch_u32(const uint32_t offset, const uint32_t val) {
  ASSERT(!detail::sInDisplayList && offset + sizeof(uint32_t) <= detail::sBufferSize,
         "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  uint32_t bytesWritten = detail::sDlWritePos;
  uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (detail::sBufferSize == 0) {
    return;
  }
  process(detail::sBufferData, detail::sBufferSize, true);
  detail::sBufferSize = 0;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
void clear_buffer() { detail::sBufferSize = 0; }

} // namespace aurora::gx::fifo
