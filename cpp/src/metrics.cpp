#include "actlab/metrics.hpp"

#include "actlab/path_utils.hpp"
#include "actlab/proc.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace actlab {
namespace {

// "1K" / "2.5M" / "123" -> 数值
double parse_big_number(const std::string& raw) {
  if (raw.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double scale = 1.0;
  std::string text = raw;
  const char suffix = text.back();
  if (suffix == 'K' || suffix == 'k') {
    scale = 1e3;
    text.pop_back();
  } else if (suffix == 'M' || suffix == 'm') {
    scale = 1e6;
    text.pop_back();
  } else if (suffix == 'B' || suffix == 'b') {
    scale = 1e9;
    text.pop_back();
  }
  try {
    return std::stod(text) * scale;
  } catch (const std::exception&) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace

bool parse_metrics_line(const std::string& line, MetricSample* out) {
  if (out == nullptr) {
    return false;
  }
  MetricSample sample;
  std::istringstream stream(line);
  std::string token;
  bool has_step = false;
  bool has_loss = false;
  while (stream >> token) {
    const size_t colon = token.find(':');
    if (colon == std::string::npos || colon == 0) {
      continue;
    }
    const std::string key = token.substr(0, colon);
    const std::string value_text = token.substr(colon + 1);
    const double value = parse_big_number(value_text);
    if (std::isnan(value)) {
      continue;
    }
    if (key == "step") {
      sample.step = static_cast<long long>(value);
      has_step = true;
    } else if (key == "smpl") {
      sample.samples = static_cast<long long>(value);
    } else if (key == "ep") {
      sample.episodes = static_cast<long long>(value);
    } else if (key == "epch") {
      sample.epochs = value;
    } else if (key == "loss") {
      sample.loss = value;
      has_loss = true;
    } else if (key == "grdn") {
      sample.grad_norm = value;
    } else if (key == "lr") {
      sample.lr = value;
    } else if (key == "step_s") {
      sample.step_s = value;
    } else if (key == "smp/s") {
      sample.samples_per_s = value;
    } else if (key == "mem_gb") {
      sample.gpu_mem_gb = value;
    }
  }
  if (!has_step || !has_loss) {
    return false;
  }
  *out = sample;
  return true;
}

std::vector<MetricSample> parse_metrics_log(const std::filesystem::path& log_path) {
  std::ifstream in(log_path);
  if (!in) {
    throw std::runtime_error("metrics: cannot open log " + log_path.string());
  }
  std::vector<MetricSample> samples;
  std::string line;
  while (std::getline(in, line)) {
    MetricSample sample;
    if (parse_metrics_line(line, &sample)) {
      samples.push_back(sample);
    }
  }
  return samples;
}

void write_metrics_jsonl(const std::vector<MetricSample>& samples, const std::filesystem::path& out_path) {
  ensure_dir(out_path.parent_path());
  std::ofstream out(out_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("metrics: cannot write " + out_path.string());
  }
  for (const auto& sample : samples) {
    out << "{\"step\": " << sample.step << ", \"samples\": " << sample.samples
        << ", \"episodes\": " << sample.episodes << ", \"epochs\": " << sample.epochs
        << ", \"loss\": " << sample.loss << ", \"grad_norm\": " << sample.grad_norm
        << ", \"lr\": " << sample.lr << ", \"step_s\": " << sample.step_s
        << ", \"samples_per_s\": " << sample.samples_per_s << ", \"gpu_mem_gb\": " << sample.gpu_mem_gb
        << "}\n";
  }
}

void write_metrics_csv(const std::vector<MetricSample>& samples, const std::filesystem::path& out_path) {
  ensure_dir(out_path.parent_path());
  std::ofstream out(out_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("metrics: cannot write " + out_path.string());
  }
  out << "step,samples,episodes,epochs,loss,grad_norm,lr,step_s,samples_per_s,gpu_mem_gb\n";
  for (const auto& sample : samples) {
    out << sample.step << "," << sample.samples << "," << sample.episodes << "," << sample.epochs << ","
        << sample.loss << "," << sample.grad_norm << "," << sample.lr << "," << sample.step_s << ","
        << sample.samples_per_s << "," << sample.gpu_mem_gb << "\n";
  }
}

std::filesystem::path write_loss_curve(const std::vector<MetricSample>& samples,
                                       const std::filesystem::path& out_dir,
                                       std::string* note) {
  ensure_dir(out_dir);
  const std::filesystem::path svg_path = out_dir / "loss_curve.svg";
  const std::filesystem::path png_path = out_dir / "loss_curve.png";

  constexpr int kWidth = 960;
  constexpr int kHeight = 540;
  constexpr int kLeft = 80;
  constexpr int kRight = 40;
  constexpr int kTop = 40;
  constexpr int kBottom = 60;

  std::ofstream svg(svg_path, std::ios::trunc);
  if (!svg) {
    throw std::runtime_error("metrics: cannot write " + svg_path.string());
  }
  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << kWidth << "\" height=\"" << kHeight
      << "\" viewBox=\"0 0 " << kWidth << " " << kHeight << "\">\n";
  svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  svg << "<text x=\"" << kLeft << "\" y=\"24\" font-family=\"monospace\" font-size=\"16\">"
      << "ACT training loss (real log data, n=" << samples.size() << ")</text>\n";

  const int plot_w = kWidth - kLeft - kRight;
  const int plot_h = kHeight - kTop - kBottom;
  svg << "<rect x=\"" << kLeft << "\" y=\"" << kTop << "\" width=\"" << plot_w << "\" height=\"" << plot_h
      << "\" fill=\"none\" stroke=\"black\"/>\n";

  if (samples.size() >= 2) {
    const long long step_min = samples.front().step;
    const long long step_max = samples.back().step;
    double loss_min = std::numeric_limits<double>::max();
    double loss_max = std::numeric_limits<double>::lowest();
    for (const auto& sample : samples) {
      loss_min = std::min(loss_min, sample.loss);
      loss_max = std::max(loss_max, sample.loss);
    }
    const double loss_span = (loss_max - loss_min) < 1e-9 ? 1.0 : (loss_max - loss_min);
    const double step_span = static_cast<double>(std::max<long long>(1, step_max - step_min));

    std::string points;
    for (const auto& sample : samples) {
      const double fx = static_cast<double>(sample.step - step_min) / step_span;
      const double fy = (sample.loss - loss_min) / loss_span;
      const int x = kLeft + static_cast<int>(fx * plot_w);
      const int y = kTop + plot_h - static_cast<int>(fy * plot_h);
      points += std::to_string(x) + "," + std::to_string(y) + " ";
    }
    svg << "<polyline fill=\"none\" stroke=\"#c0392b\" stroke-width=\"2\" points=\"" << points << "\"/>\n";
    svg << "<text x=\"" << kLeft << "\" y=\"" << (kTop + plot_h + 24)
        << "\" font-family=\"monospace\" font-size=\"13\">step " << step_min << " → " << step_max
        << "</text>\n";
    svg << "<text x=\"" << kLeft << "\" y=\"" << (kTop + plot_h + 44)
        << "\" font-family=\"monospace\" font-size=\"13\">loss " << loss_min << " → " << loss_max
        << "</text>\n";
  } else {
    svg << "<text x=\"" << (kLeft + 20) << "\" y=\"" << (kTop + 40)
        << "\" font-family=\"monospace\" font-size=\"14\">NOT_ENOUGH_DATA</text>\n";
  }
  svg << "</svg>\n";
  svg.close();

  const std::string convert_bin = find_executable("convert");
  if (convert_bin.empty()) {
    if (note != nullptr) {
      *note = "ImageMagick 'convert' not found; SVG kept, PNG not produced";
    }
    return svg_path;
  }
  const ProcResult result = run_process({convert_bin, "-density", "96", svg_path.string(), png_path.string()},
                                        ProcOptions{});
  if (result.exit_code != 0 || !std::filesystem::exists(png_path)) {
    if (note != nullptr) {
      *note = "convert failed with exit code " + std::to_string(result.exit_code);
    }
    return svg_path;
  }
  if (note != nullptr) {
    *note = "PNG rendered from real log data via ImageMagick";
  }
  return png_path;
}

}  // namespace actlab
