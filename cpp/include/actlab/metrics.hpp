// 训练日志 → 指标聚合：解析 MetricsTracker 行、写 JSONL/CSV、生成真实 loss 曲线
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace actlab {

// 对应 LeRobot utils/logging_utils.py MetricsTracker.__str__ 的输出，例如：
//   step:1K smpl:8K ep:3 epch:0.10 loss:0.123 grdn:1.23 lr:1.0e-05 step_s:0.20 smp/s:40 mem_gb:3.21
struct MetricSample {
  long long step = -1;
  long long samples = -1;
  long long episodes = -1;
  double epochs = -1.0;
  double loss = -1.0;
  double grad_norm = -1.0;
  double lr = -1.0;
  double step_s = -1.0;
  double samples_per_s = -1.0;
  double gpu_mem_gb = -1.0;
};

// 解析单行；识别不到 step 或 loss 时返回 false（其余字段保留 -1 语义）
bool parse_metrics_line(const std::string& line, MetricSample* out);

// 解析整个训练日志
std::vector<MetricSample> parse_metrics_log(const std::filesystem::path& log_path);

// 写 JSONL（每行一个样本）
void write_metrics_jsonl(const std::vector<MetricSample>& samples, const std::filesystem::path& out_path);

// 写 loss/step/吞吐 CSV（表头固定）
void write_metrics_csv(const std::vector<MetricSample>& samples, const std::filesystem::path& out_path);

// 依据真实样本生成 SVG 折线图（loss + step_s），再用 ImageMagick 栅格化为 PNG。
// 返回 PNG 路径；栅格化工具缺失时返回 SVG 路径并在 error 中说明。
std::filesystem::path write_loss_curve(const std::vector<MetricSample>& samples,
                                       const std::filesystem::path& out_dir,
                                       std::string* note);

}  // namespace actlab
