#include "actlab/npy_io.hpp"

#include "actlab/path_utils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace actlab {
namespace {

constexpr char kMagic[] = "\x93NUMPY";

std::map<std::string, int> kDtypeSizes = {{"<u1", 1}, {"|u1", 1}, {"<f4", 4}, {"<f8", 8},
                                          {"<i8", 8}, {"<i4", 4}, {"<f2", 2}};

}  // namespace

long long NpyArray::element_count() const {
  long long count = 1;
  for (const auto dim : shape) {
    count *= dim;
  }
  return shape.empty() ? 0 : count;
}

long long NpyArray::bytes_per_element() const {
  const auto it = kDtypeSizes.find(dtype);
  if (it == kDtypeSizes.end()) {
    throw std::runtime_error("npy: unsupported dtype '" + dtype + "'");
  }
  return it->second;
}

NpyArray read_npy(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("npy: cannot open " + path.string());
  }
  char magic[6] = {};
  in.read(magic, 6);
  if (std::memcmp(magic, kMagic, 6) != 0) {
    throw std::runtime_error("npy: bad magic in " + path.string());
  }
  uint8_t major = 0;
  uint8_t minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);
  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t short_len = 0;
    in.read(reinterpret_cast<char*>(&short_len), 2);
    header_len = short_len;
  } else {
    in.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));

  const size_t descr_pos = header.find("'descr'");
  const size_t descr_quote = header.find('\'', header.find(':', descr_pos) + 1);
  const size_t descr_end = header.find('\'', descr_quote + 1);
  if (descr_pos == std::string::npos || descr_quote == std::string::npos || descr_end == std::string::npos) {
    throw std::runtime_error("npy: cannot parse 'descr' in " + path.string());
  }
  NpyArray array;
  array.dtype = header.substr(descr_quote + 1, descr_end - descr_quote - 1);

  const size_t shape_pos = header.find("'shape'");
  const size_t shape_open = header.find('(', shape_pos);
  const size_t shape_close = header.find(')', shape_open);
  if (shape_pos == std::string::npos || shape_open == std::string::npos || shape_close == std::string::npos) {
    throw std::runtime_error("npy: cannot parse 'shape' in " + path.string());
  }
  std::istringstream shape_stream(header.substr(shape_open + 1, shape_close - shape_open - 1));
  std::string token;
  while (std::getline(shape_stream, token, ',')) {
    const size_t begin = token.find_first_not_of(" \t");
    if (begin == std::string::npos) {
      continue;
    }
    const size_t end = token.find_last_not_of(" \t");
    array.shape.push_back(std::stoll(token.substr(begin, end - begin + 1)));
  }

  const long long elements = array.element_count();
  const long long bytes = elements * array.bytes_per_element();
  array.data.resize(static_cast<size_t>(bytes));
  in.read(reinterpret_cast<char*>(array.data.data()), bytes);
  if (in.gcount() != bytes) {
    throw std::runtime_error("npy: truncated data in " + path.string());
  }
  return array;
}

void write_ppm(const uint8_t* rgb, long long height, long long width, const std::filesystem::path& path) {
  ensure_dir(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("ppm: cannot write " + path.string());
  }
  out << "P6\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(rgb), height * width * 3);
}

void write_montage_ppm(const std::vector<NpyArray>& images, const std::filesystem::path& path, int columns,
                       int scale) {
  if (images.empty()) {
    throw std::runtime_error("montage: no images");
  }
  const NpyArray& first = images.front();
  if (first.shape.size() != 3 || first.shape[2] != 3 ||
      (first.dtype != "<u1" && first.dtype != "|u1")) {
    throw std::runtime_error("montage: expected HWC uint8 RGB arrays");
  }
  const long long height = first.shape[0];
  const long long width = first.shape[1];
  for (const auto& image : images) {
    if (image.shape.size() != 3 || image.shape[0] != height || image.shape[1] != width) {
      throw std::runtime_error("montage: inconsistent image shapes");
    }
  }

  const int cols = columns > 0 ? columns : 1;
  const int rows = static_cast<int>((images.size() + static_cast<size_t>(cols) - 1) / static_cast<size_t>(cols));
  const int tile_w = static_cast<int>(width) / scale;
  const int tile_h = static_cast<int>(height) / scale;
  const int out_w = tile_w * cols;
  const int out_h = tile_h * rows;
  std::vector<uint8_t> canvas(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3, 32);

  for (size_t index = 0; index < images.size(); ++index) {
    const int tile_col = static_cast<int>(index) % cols;
    const int tile_row = static_cast<int>(index) / cols;
    const NpyArray& image = images[index];
    for (int y = 0; y < tile_h; ++y) {
      for (int x = 0; x < tile_w; ++x) {
        const long long src_y = static_cast<long long>(y) * scale;
        const long long src_x = static_cast<long long>(x) * scale;
        const uint8_t* src = image.data.data() + (src_y * width + src_x) * 3;
        uint8_t* dst = canvas.data() + ((static_cast<size_t>(tile_row) * tile_h + y) * out_w +
                                        static_cast<size_t>(tile_col) * tile_w + x) *
                                          3;
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
      }
    }
  }
  write_ppm(canvas.data(), out_h, out_w, path);
}

std::vector<uint8_t> float_chw_to_uint8_hwc(const NpyArray& image) {
  if (image.shape.size() != 3 || image.shape[0] != 3) {
    throw std::runtime_error("float_chw_to_uint8_hwc: expected CHW array with 3 channels");
  }
  if (image.dtype != "<f4" && image.dtype != "<f8") {
    throw std::runtime_error("float_chw_to_uint8_hwc: expected float32/float64, got " + image.dtype);
  }
  const long long height = image.shape[1];
  const long long width = image.shape[2];
  const bool is_double = image.dtype == "<f8";
  std::vector<uint8_t> out(static_cast<size_t>(height * width * 3));
  for (long long y = 0; y < height; ++y) {
    for (long long x = 0; x < width; ++x) {
      for (long long channel = 0; channel < 3; ++channel) {
        const size_t src_index = static_cast<size_t>((channel * height + y) * width + x);
        double value = 0.0;
        if (is_double) {
          double raw = 0.0;
          std::memcpy(&raw, image.data.data() + src_index * sizeof(double), sizeof(double));
          value = raw;
        } else {
          float raw = 0.0f;
          std::memcpy(&raw, image.data.data() + src_index * sizeof(float), sizeof(float));
          value = static_cast<double>(raw);
        }
        value = std::min(1.0, std::max(0.0, value));
        out[static_cast<size_t>((y * width + x) * 3 + channel)] =
            static_cast<uint8_t>(value * 255.0 + 0.5);
      }
    }
  }
  return out;
}

}  // namespace actlab
