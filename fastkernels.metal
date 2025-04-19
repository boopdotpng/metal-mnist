#include <metal_stdlib>
using namespace metal;

// parameter struct 
struct dims_t { uint batch, din, dout; };

// forward pass kernels
kernel void matmul_bias_relu(
  device float *input [[ buffer(0) ]],
  device float *w [[ buffer(1) ]],
  device float *bias [[ buffer(2) ]],
  device char *relu_mask [[ buffer(3) ]],
  device float *output [[ buffer(4) ]],
  constant dims_t &dims [[ buffer(5) ]],
  uint id [[thread_position_in_grid]]
) {
  uint b = id / dims.dout;
  uint j = id % dims.dout;

  float sum = 0.0f;
  for (uint i = 0; i < dims.din; ++i) {
    float x_val = input[b * dims.din + i];
    float w_val = w[i * dims.dout + j];
    sum += x_val * w_val;
  }

  sum += bias[j];
  output[b * dims.dout + j] = max(0.0f, sum);  // ReLU
  relu_mask[b * dims.dout + j] = sum > 0 ? 1 : 0;
}

kernel void matmul_bias(
  device const float   *input  [[ buffer(0) ]],
  device const float   *w      [[ buffer(1) ]],
  device const float   *bias   [[ buffer(2) ]],
  device       float   *output [[ buffer(3) ]],
  constant     dims_t  &dims   [[ buffer(4) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint b = id / dims.dout;
  uint j = id % dims.dout;
  if (b >= dims.batch) return;

  float sum = 0.0f;
  for (uint i = 0; i < dims.din; ++i)
    sum += input[b * dims.din + i] * w[i * dims.dout + j];

  sum += bias[j];
  output[b * dims.dout + j] = sum;
}


// cross_entropy: use dims.dout for classes, write loss_buf
kernel void cross_entropy(
  device const float *logits   [[ buffer(0) ]],
  device const uchar *labels   [[ buffer(1) ]],
  device       float *loss_buf [[ buffer(2) ]],
  constant     dims_t &dims     [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint b = id;
  if (b >= dims.batch) return;
  uint C = dims.dout;
  device const float* row = logits + b * C;

  float max_val = row[0];
  for (uint j = 1; j < C; ++j)
    max_val = fmax(row[j], max_val);

  float denom = 0.0f;
  for (uint j = 0; j < C; ++j)
    denom += exp(row[j] - max_val);

  uchar lab = labels[b];
  float log_prob = row[lab] - max_val - log(denom);
  loss_buf[b] = -log_prob;
}


// cross_entropy_backward: same dims, write dL_dlogits
kernel void cross_entropy_backward(
  device const float *logits      [[ buffer(0) ]],
  device const uchar *labels      [[ buffer(1) ]],
  device       float *dL_dlogits  [[ buffer(2) ]],
  constant     dims_t &dims        [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint C = dims.dout;
  uint idx = id;
  uint b = idx / C;
  uint j = idx % C;
  if (b >= dims.batch) return;

  device const float* row = logits + b * C;
  // find max
  float m = row[0];
  for (uint k = 1; k < C; ++k) m = fmax(row[k], m);

  float denom = 0.0f;
  for (uint k = 0; k < C; ++k)
    denom += exp(row[k] - m);

  float prob = exp(row[j] - m) / denom;
  if (j == labels[b]) prob -= 1.0f;
  dL_dlogits[b * C + j] = prob / float(dims.batch);
}


kernel void matmul_grad_w (
  device float *input [[ buffer(0) ]],
  device float *dL_dout [[ buffer(1) ]],
  device float *dW [[ buffer(2) ]],
  constant dims_t &dims [[ buffer(3) ]], // batch, d_in, d_out
  uint id [[ thread_position_in_grid ]]
) {
  uint i = id / dims.dout;
  uint j = id % dims.dout;

  float sum = 0.0f;
  for (uint b = 0; b < dims.batch; b++) {
    float x_val = input[b * dims.din + i];
    float grad = dL_dout[b * dims.dout + j];
    sum += x_val * grad;
  }

  dW[i * dims.dout + j] = sum;
}

kernel void relu_backward(
  device const char *relu_mask [[ buffer(0) ]],
  device       float *dL_dout   [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]]
) {
  if (relu_mask[id] == 0) dL_dout[id] = 0.0f;
}


kernel void bias_grad_sum (
  device const float* dL_dout [[ buffer(0) ]],  // shape: (B, Dout)
  device float* dB [[ buffer(1) ]],             // shape: (Dout,)
  constant dims_t &dims [[ buffer(2) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint B = dims.batch;
  uint Dout = dims.dout;
  uint j = id;
  if (j >= Dout) return;

  float sum = 0.0f;
  for (uint b = 0; b < B; ++b) {
    sum += dL_dout[b * Dout + j];
  }

  dB[j] = sum;
}

kernel void matmul_grad_input(
  device const float* dL_dout [[ buffer(0) ]],   // shape: (B, Dout)
  device const float* W [[ buffer(1) ]],         // shape: (Din, Dout)
  device float* dL_dx [[ buffer(2) ]],           // shape: (B, Din)
  constant dims_t& dims [[ buffer(3) ]],
  uint id [[ thread_position_in_grid ]]
) {
  uint B = dims.batch;
  uint Din = dims.din;
  uint Dout = dims.dout;

  uint b = id / Din;  // which sample
  uint i = id % Din;  // which input dim

  if (b >= B) return;

  float sum = 0.0f;
  for (uint j = 0; j < Dout; ++j) {
    float d_out = dL_dout[b * Dout + j];
    float w_ij = W[i * Dout + j]; // W is [Din][Dout]
    sum += d_out * w_ij;
  }

  dL_dx[b * Din + i] = sum;
}

/*
launch one thread per weight that needs to be updated
*/
kernel void sgd_update(
  device float* p [[buffer(0)]],
  device float *g [[ buffer(1) ]],
  const device float* lr[[buffer(2)]],
  uint id [[thread_position_in_grid]]
) {
  p[id] -= lr[0] * g[id];
}

/*
launch one thread per weight that needs to be updated
*/
kernel void normalize_image(
  device const uchar *in [[ buffer(0) ]],
  device float *out [[ buffer(1) ]],
  uint id [[ thread_position_in_grid ]] 
) {
  out[id] = (float)(in[id]) / 255.0f;
}

// layer init 
// TODO: actually launch this kernel in the code
kernel void fill_uniform(
  device float *x [[ buffer(0) ]],
  uint id [[ thread_position_in_grid ]]
) {
  x[id] = 0; // DO THIS LATER! 
}

// misc. 
kernel void zero_buffer(
  device float *x [[ buffer(0) ]],
  uint id [[ thread_position_in_grid ]]
) {
  x[id] = 0;
}
