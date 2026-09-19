// 极简 .npy 读写（纯 C++，不依赖 torch）：供数据集样本落盘与 C++ 侧推理输入读取
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace actlab {

struct NpyArray {
  std::vector<long long> shape;
  std::string dtype;                 // "<u1" / "<f4" / "<f8" / "<i8"
  std::vector<uint8_t> data;         // 原始 C 顺序字节

  long long element_count() const;
  long long bytes_per_element() const;
};

NpyArray read_npy(const std::filesystem::path& path);

// 以 P6 二进制 PPM 写单张 HWC uint8 图像（随后可由 ImageMagick 转 PNG）
void write_ppm(const uint8_t* rgb, long long height, long long width, const std::filesystem::path& path);

// 生成横向拼接的样本拼图（每张图等尺寸、HWC uint8）
void write_montage_ppm(const std::vector<NpyArray>& images, const std::filesystem::path& path,
                       int columns = 4, int scale = 1);

// 把 float32/float64 CHW [0,1] 图像转成 HWC uint8（仅用于证据拼图，不参与模型前向）
std::vector<uint8_t> float_chw_to_uint8_hwc(const NpyArray& image);

}  // namespace actlab
