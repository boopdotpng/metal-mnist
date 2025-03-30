#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
  uint M; // batch size
  uint K; // input dim
  uint N; // output dim
};
struct InitParams {uint32_t param_count, fan_in, fan_out; };

kernel void copy_batch(
  device const float *src [[ buffer(0) ]],
  device float *dest [[ buffer(1) ]],
  constant uint &offset [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  dest[id] = src[offset * 784 + id];
}

kernel void copy_batch_labels(
  device const uchar *src [[ buffer(0) ]],
  device uchar *dest [[ buffer(1) ]],
  constant uint &offset [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  dest[id] = src[offset + id];
}

kernel void normalize_image(
  device const uchar *in [[ buffer(0) ]],
  device float *out [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]],
  uint total_threads [[ threads_per_grid ]]
) {
  for (uint i = id; i < 47040000; i += total_threads) {
    out[i] = (float)(in[i]) / 255.0f;
  }
}

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
  uint rng = lcg(id ^ 1234);
  
  for (uint i = id; i < params.param_count; i += total_threads) {
    rng = lcg(rng);
    float r = (rng & 0x00FFFFFF) / 16777216.0f;
    weights[i] = (r * 2.0f - 1.0f) * bound;
  }
}

kernel void relu(
  device float *weights [[ buffer(0) ]],
  uint id [[ thread_position_in_grid ]]
) {
    weights[id] = max(weights[id], 0.0f);
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

  if (id >= M * N) return;

  uint i = id / N; // row index
  int j = id % N;  // col index

  float sum = 0.0f;
  for (uint k = 0; k < K; ++k) {
    float a_ik = a[i * K + k];
    float b_kj = b[k * N + j];
    sum += a_ik * b_kj;
  }
  c[i * N + j] = sum;
}

kernel void add_bias(
  device float *in_out [[ buffer(0) ]],
  device const float *bias [[ buffer(1) ]],
  constant uint &dim [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  in_out[id] += bias[id % dim];
}

kernel void softmax(
  device const float *logits [[ buffer(0) ]],
  device float *probs [[ buffer(1) ]],
  constant uint &output_dim [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  // figure out which row we're in
  uint row = id / output_dim;
  uint col = id % output_dim;
  uint row_start = row * output_dim;

  // first: find max for numerical stability
  float maxval = logits[row_start];
  for (uint i = 1; i < output_dim; ++i) {
    float v = logits[row_start + i];
    if (v > maxval) maxval = v;
  }

  // compute exp(x - max)
  float denom = 0.0f;
  for (uint i = 0; i < output_dim; ++i) {
    denom += exp(logits[row_start + i] - maxval);
  }

  // final softmax output
  probs[id] = exp(logits[id] - maxval) / denom;
}

kernel void cross_entropy_loss(
  device const float *softmax_output [[ buffer(0) ]],
  device const uchar *labels [[ buffer(1) ]],
  device float *losses [[ buffer(2) ]], // per-sample losses
  constant uint &num_classes [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint label = labels[id];
  float prob = softmax_output[id * num_classes + label];
  losses[id] = -log(prob + 1e-8f); // prevent log(0)
}

kernel void softmax_cross_entropy_backward(
  device const float *softmax [[ buffer(0) ]],     // y_hat
  device const uchar *labels [[ buffer(1) ]],      // y_true
  device float *dL_dY [[ buffer(2) ]],             // output
  constant uint &num_classes [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint row = id / num_classes;
  uint col = id % num_classes;
  uint label = labels[row];

  float grad = softmax[id];
  if (col == label) grad -= 1.0f;

  dL_dY[id] = grad;
}

// grad_w: W_grad = Aᵗ @ dL_dY
kernel void matmul_grad_w(
  device const float *A [[ buffer(0) ]],        // activations from layer 2
  device const float *dL_dY [[ buffer(1) ]],    // softmax gradient
  device float *gradW [[ buffer(2) ]],          // output: grad weights
  constant MatmulParams &shape [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint K = shape.K;  // input dim = 128
  uint N = shape.N;  // output dim = 10
  uint M = shape.M;  // batch size

  if (id >= K * N) return;

  uint k = id / N;
  uint n = id % N;

  float sum = 0.0f;
  for (uint m = 0; m < M; ++m) {
    float a_mk = A[m * K + k];
    float dy_mn = dL_dY[m * N + n];
    sum += a_mk * dy_mn;
  }

  gradW[k * N + n] = sum / float(M);
}

// grad_input: dA = dY @ Wᵗ
kernel void matmul_grad_input(
  device const float *dY [[ buffer(0) ]],     // [B × N]
  device const float *W [[ buffer(1) ]],      // [K × N]
  device float *dA [[ buffer(2) ]],           // [B × K]
  constant MatmulParams &shape [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint M = shape.M;  // batch size
  uint K = shape.K;  // input dim (128)
  uint N = shape.N;  // output dim (10)

  if (id >= M * K) return;

  uint m = id / K;  // batch index
  uint k = id % K;  // input dim index

  float sum = 0.0f;
  for (uint n = 0; n < N; ++n) {
    float dy_mn = dY[m * N + n];
    float w_kn = W[k * N + n];
    sum += dy_mn * w_kn;
  }

  dA[m * K + k] = sum;
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
  device const float *activation [[ buffer(0) ]],  
  device float *grad [[ buffer(1) ]],             
  uint id [[ thread_position_in_grid ]]
) {
  if (activation[id] <= 0.0f) {
    grad[id] = 0.0f;
  }
}

kernel void bias_grad_sum(
  device const float *dL_dA [[ buffer(0) ]],
  device float *grad_bias [[ buffer(1) ]],
  constant uint &batch_size [[ buffer(2) ]],
  constant uint &dim [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  float sum = 0.0f;
  for (uint i = 0; i < batch_size; ++i) {
    sum += dL_dA[i * dim + id];
  }
  grad_bias[id] = sum / float(batch_size);
}
