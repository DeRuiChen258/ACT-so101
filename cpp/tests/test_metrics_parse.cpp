#include "actlab/metrics.hpp"

#include "actlab_test.hpp"

#include <filesystem>
#include <fstream>

ACTLAB_TEST(metrics_parse_realistic_line) {
  const std::string line =
      "step:1K smpl:8K ep:3 epch:0.10 loss:0.123 grdn:1.230 lr:1.0e-05 data_s:0.010 prep_s:0.020 "
      "updt_s:0.100 step_s:0.200 smp/s:40 mem_gb:3.21";
  actlab::MetricSample sample;
  CHECK_TRUE(actlab::parse_metrics_line(line, &sample));
  CHECK_EQ(sample.step, 1000LL);
  CHECK_EQ(sample.samples, 8000LL);
  CHECK_EQ(sample.episodes, 3LL);
  CHECK_TRUE(sample.loss > 0.122 && sample.loss < 0.124);
  CHECK_TRUE(sample.samples_per_s > 39.0 && sample.samples_per_s < 41.0);
  CHECK_TRUE(sample.gpu_mem_gb > 3.2 && sample.gpu_mem_gb < 3.22);
}

ACTLAB_TEST(metrics_parse_rejects_noise) {
  actlab::MetricSample sample;
  CHECK_TRUE(!actlab::parse_metrics_line("Start offline training on a fixed dataset", &sample));
  CHECK_TRUE(!actlab::parse_metrics_line("loss:0.5 only", &sample));
}

ACTLAB_TEST(metrics_log_and_csv_roundtrip) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "actlab_metrics_test";
  std::filesystem::create_directories(dir);
  const std::filesystem::path log = dir / "train.log";
  {
    std::ofstream out(log, std::ios::trunc);
    out << "noise line\n";
    out << "step:10 loss:1.0 step_s:0.5\n";
    out << "step:20 loss:0.5 step_s:0.4\n";
  }
  const auto samples = actlab::parse_metrics_log(log);
  CHECK_EQ(samples.size(), static_cast<size_t>(2));
  const std::filesystem::path csv = dir / "loss_curve.csv";
  actlab::write_metrics_csv(samples, csv);
  CHECK_TRUE(std::filesystem::exists(csv));
  CHECK_TRUE(std::filesystem::file_size(csv) > 40);
  std::filesystem::remove_all(dir);
}
