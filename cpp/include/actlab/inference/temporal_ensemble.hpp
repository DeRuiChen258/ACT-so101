// Temporal Ensembling：与 LeRobot ACTTemporalEnsembler 的在线实现保持一致的指数加权融合
// 参考实现：src/lerobot/policies/act/modeling_act.py::ACTTemporalEnsembler
//   w_i = exp(-coeff * i)（i 为动作块内的年龄索引，i=0 最旧），在线更新等价于离线加权平均。
#pragma once

#include <torch/torch.h>

#include <deque>

namespace actlab {

class TemporalEnsembler {
 public:
  TemporalEnsembler(double coefficient, int64_t chunk_size);

  // 输入 (batch, chunk_size, action_dim) 的动作块，返回融合后的 (batch, action_dim) 下一步动作
  torch::Tensor update(const torch::Tensor& action_chunk);

  void reset();
  bool active() const { return ensembled_actions_.defined(); }
  int64_t chunk_size() const { return chunk_size_; }

 private:
  double coefficient_;
  int64_t chunk_size_;
  torch::Tensor weights_;
  torch::Tensor weights_cumsum_;
  torch::Tensor ensembled_actions_;
  torch::Tensor ensembled_actions_count_;
  std::deque<torch::Tensor> dummy_;  // 保持接口形态（与 Python 端行为对齐的占位队列）
};

}  // namespace actlab
