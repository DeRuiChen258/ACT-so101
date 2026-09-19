#include "actlab/inference/temporal_ensemble.hpp"

#include <stdexcept>

namespace actlab {

TemporalEnsembler::TemporalEnsembler(double coefficient, int64_t chunk_size)
    : coefficient_(coefficient), chunk_size_(chunk_size) {
  if (chunk_size <= 0) {
    throw std::runtime_error("TemporalEnsembler: chunk_size must be > 0");
  }
  const auto index = torch::arange(chunk_size, torch::TensorOptions().dtype(torch::kFloat32));
  weights_ = torch::exp(-coefficient * index);
  weights_cumsum_ = torch::cumsum(weights_, 0);
  reset();
}

void TemporalEnsembler::reset() {
  ensembled_actions_ = torch::Tensor();
  ensembled_actions_count_ = torch::Tensor();
}

torch::Tensor TemporalEnsembler::update(const torch::Tensor& action_chunk) {
  if (action_chunk.dim() != 3) {
    throw std::runtime_error("TemporalEnsembler::update expects (batch, chunk_size, action_dim)");
  }
  const int64_t batch = action_chunk.size(0);
  const int64_t chunk = action_chunk.size(1);
  const int64_t action_dim = action_chunk.size(2);
  const auto weights = weights_.to(action_chunk.device());
  const auto weights_cumsum = weights_cumsum_.to(action_chunk.device());

  if (!ensembled_actions_.defined()) {
    ensembled_actions_ = action_chunk.clone();
    ensembled_actions_count_ =
        torch::ones({chunk, 1}, action_chunk.options().dtype(torch::kLong));
  } else {
    // 与 Python 在线实现逐行对应：
    //   ensembled *= w_cumsum[count-1]; ensembled += actions[:, :-1] * w[count];
    //   ensembled /= w_cumsum[count];  count = clamp(count+1, max=chunk)
    const auto previous_count = ensembled_actions_count_;
    ensembled_actions_ = ensembled_actions_ * weights_cumsum.index({previous_count - 1});
    ensembled_actions_ = ensembled_actions_ +
        action_chunk.slice(1, 0, chunk - 1) * weights.index({previous_count});
    ensembled_actions_ = ensembled_actions_ / weights_cumsum.index({previous_count});
    ensembled_actions_count_ = torch::clamp(previous_count + 1, c10::nullopt, chunk_size_);
    ensembled_actions_ =
        torch::cat({ensembled_actions_, action_chunk.slice(1, chunk - 1, chunk)}, 1);
    ensembled_actions_count_ = torch::cat(
        {ensembled_actions_count_,
         torch::ones({1, 1}, ensembled_actions_count_.options())},
        0);
  }
  auto action = ensembled_actions_.slice(1, 0, 1).squeeze(1);
  ensembled_actions_ = ensembled_actions_.slice(1, 1, ensembled_actions_.size(1));
  ensembled_actions_count_ = ensembled_actions_count_.slice(0, 1, ensembled_actions_count_.size(0));
  (void)batch;
  (void)action_dim;
  return action;
}

}  // namespace actlab
