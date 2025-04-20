#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
  uint M; // batch size
  uint K; // input dim
  uint N; // output dim
};
struct InitParams { uint32_t param_count, fan_in, fan_out; };

kernel void copy_batch(
  device const float *src [[ buffer(0) ]],
  device float *dest [[ buffer(1) ]],
  constant uint &offset [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  // assumes batch data is contiguous in src after offset
  // id = thread index within the batch copy (0 to batch_size * input_dim - 1)
  // input_dim is implicitly 784 here.
  uint src_idx = offset * 784 + id;
  dest[id] = src[src_idx];
}

kernel void copy_batch_labels(
  device const uchar *src [[ buffer(0) ]],
  device uchar *dest [[ buffer(1) ]],
  constant uint &offset [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  // id = thread index within the label copy (0 to batch_size - 1)
  uint src_idx = offset + id;
  dest[id] = src[src_idx];
}

kernel void normalize_image(
  device const uchar *in [[ buffer(0) ]],
  device float *out [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]] // use thread position directly
) {
  // no loop needed if grid size == data size
  out[id] = (float)(in[id]) / 255.0f;
}

// --- weight initialization ---
inline uint lcg(uint x) {
  return x * 1664525u + 1013904223u;
}

inline float rand_uniform(uint id, uint seed) {
  return (float)(lcg(id ^ seed) & 0x00FFFFFF) / 16777216.0f;
}

kernel void init_random_weights(
  device float *weights [[ buffer(0) ]],
  constant InitParams &params [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]],
  uint total_threads [[ threads_per_grid ]]
) {
  const float bound = sqrt(6.0f / (params.fan_in + params.fan_out));
  uint rng = lcg(id ^ 1234); // reproducible fixed seed + id

  // grid-stride loop
  for (uint i = id; i < params.param_count; i += total_threads) {
    rng = lcg(rng);
    float r = (rng & 0x00FFFFFF) / 16777216.0f;
    weights[i] = (r * 2.0f - 1.0f) * bound;
  }
}
// --- end weight initialization ---

// modified relu: takes input and output buffers
kernel void relu(
  device const float *in_act [[ buffer(0) ]],
  device float *out_act [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]]
) {
  out_act[id] = max(in_act[id], 0.0f);
}

kernel void matmul(
  device const float *a [[ buffer(0) ]],
  device const float *b [[ buffer(1) ]],
  device float *c [[ buffer(2) ]],
  constant MatmulParams &params [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint M = params.M;
  uint K = params.K;
  uint N = params.N;

  uint i = id / N; // row index
  uint j = id % N; // col index
  if (i >= M) return;

  float sum = 0.0f;
  for (uint k = 0; k < K; ++k) {
    uint a_idx = i * K + k;
    uint b_idx = k * N + j;
    sum += a[a_idx] * b[b_idx];
  }
  uint c_idx = i * N + j;
  c[c_idx] = sum;
}

kernel void add_bias(
  device float *in_out [[ buffer(0) ]],
  device const float *bias [[ buffer(1) ]],
  constant uint &dim [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint bias_idx = id % dim;
  in_out[id] += bias[bias_idx];
}

kernel void softmax(
  device const float *logits [[ buffer(0) ]],
  device float *probs [[ buffer(1) ]],
  constant uint &output_dim [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint row = id / output_dim;
  uint col = id % output_dim;
  uint row_start = row * output_dim;

  float maxval = -INFINITY;
  for (uint i = 0; i < output_dim; ++i) {
    maxval = max(maxval, logits[row_start + i]);
  }

  float denom = 0.0f;
  for (uint i = 0; i < output_dim; ++i) {
    denom += exp(logits[row_start + i] - maxval);
  }

  probs[id] = exp(logits[id] - maxval) / denom;
}

kernel void cross_entropy_loss(
  device const float *softmax_output [[ buffer(0) ]],
  device const uchar *labels [[ buffer(1) ]],
  device float *losses [[ buffer(2) ]],
  constant uint &num_classes [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint label = labels[id];
  uint prob_idx = id * num_classes + label;
  float prob = max(softmax_output[prob_idx], 1e-9f);
  losses[id] = -log(prob);
}

kernel void softmax_cross_entropy_backward(
  device const float *softmax_probs [[ buffer(0) ]],
  device const uchar *labels [[ buffer(1) ]],
  device float *dL_dlogits [[ buffer(2) ]],
  constant uint &num_classes [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint row = id / num_classes;
  uint col = id % num_classes;
  uint true_label = labels[row];

  float prob = softmax_probs[id];
  float target = (col == true_label) ? 1.0f : 0.0f;
  dL_dlogits[id] = prob - target;
}

kernel void matmul_grad_w(
  device const float *A [[ buffer(0) ]],
  device const float *dL_dY [[ buffer(1) ]],
  device float *gradW [[ buffer(2) ]],
  constant MatmulParams &shape [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint K = shape.K;
  uint N = shape.N;
  uint M = shape.M;

  uint k = id / N;
  uint n = id % N;

  if (k >= K) return;

  float sum = 0.0f;
  for (uint m = 0; m < M; ++m) {
    sum += A[m * K + k] * dL_dY[m * N + n];
  }
  gradW[id] = sum / float(M);
}

kernel void matmul_grad_input(
  device const float *dY [[ buffer(0) ]],
  device const float *W [[ buffer(1) ]],
  device float *dA [[ buffer(2) ]],
  constant MatmulParams &shape [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint M = shape.M;
  uint K = shape.K;
  uint N = shape.N;

  uint m = id / K;
  uint k = id % K;
  if (m >= M) return;

  float sum = 0.0f;
  for (uint n = 0; n < N; ++n) {
    sum += dY[m * N + n] * W[k * N + n];
  }
  dA[id] = sum;
}

kernel void sgd_update(
  device float *weights [[ buffer(0) ]],
  device const float *grads [[ buffer(1) ]],
  constant float &lr [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  weights[id] -= lr * grads[id];
}

kernel void relu_backward(
  device const float *pre_relu_activation [[ buffer(0) ]],
  device float *grad_in_out [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]]
) {
  if (pre_relu_activation[id] <= 0.0f) {
    grad_in_out[id] = 0.0f;
  }
}

kernel void bias_grad_sum(
  device const float *dL_dA [[ buffer(0) ]],
  device float *grad_bias [[ buffer(1) ]],
  constant uint &batch_size [[ buffer(2) ]],
  constant uint &dim [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  if (id >= dim) return;

  float sum = 0.0f;
  for (uint i = 0; i < batch_size; ++i) {
    sum += dL_dA[i * dim + id];
  }
  grad_bias[id] = sum / float(batch_size);
}
