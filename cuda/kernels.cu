// forward pass: matmul + bias + relu
__global__ void matmul_bias_relu(
    const float* input,
    const float* w,
    const float* bias,
    char*        relu_mask,
    float*       output,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    unsigned int b = id / d.dout;
    unsigned int j = id % d.dout;
    float sum = 0.0f;
    for (unsigned int i = 0; i < d.din; ++i) {
        sum += input[b * d.din + i] * w[i * d.dout + j];
    }
    sum += bias[j];
    float activated = sum > 0.0f ? sum : 0.0f;
    output[id] = activated;
    relu_mask[id] = (sum > 0.0f) ? 1 : 0;
}

// forward pass: matmul + bias
__global__ void matmul_bias(
    const float* input,
    const float* w,
    const float* bias,
    float*       output,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    unsigned int b = id / d.dout;
    unsigned int j = id % d.dout;
    float sum = 0.0f;
    for (unsigned int i = 0; i < d.din; ++i) {
        sum += input[b * d.din + i] * w[i * d.dout + j];
    }
    sum += bias[j];
    output[id] = sum;
}

// cross-entropy loss forward
__global__ void cross_entropy(
    const float*         logits,
    const unsigned char* labels,
    float*               loss_buf,
    dims                 d,
    unsigned int         N
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= d.batch) return;
    unsigned int C = d.dout;
    const float* row = logits + b * C;
    // numeric stability
    float m = row[0];
    for (unsigned int j = 1; j < C; ++j) m = fmaxf(m, row[j]);
    float sum = 0.0f;
    for (unsigned int j = 0; j < C; ++j) sum += expf(row[j] - m);
    unsigned int lab = labels[b];
    float log_prob = row[lab] - m - logf(sum);
    loss_buf[b] = -log_prob;
}

// cross-entropy backward
__global__ void cross_entropy_backward(
    const float*         logits,
    const unsigned char* labels,
    float*               dL_dlogits,
    dims                 d,
    unsigned int         N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = d.batch * d.dout;
    if (idx >= total) return;
    unsigned int b = idx / d.dout;
    unsigned int j = idx % d.dout;
    const float* row = logits + b * d.dout;
    // numeric stability
    float m = row[0];
    for (unsigned int k = 1; k < d.dout; ++k) m = fmaxf(m, row[k]);
    float sum = 0.0f;
    for (unsigned int k = 0; k < d.dout; ++k) sum += expf(row[k] - m);
    float prob = expf(row[j] - m) / sum;
    unsigned int lab = labels[b];
    if (j == lab) prob -= 1.0f;
    dL_dlogits[idx] = prob;
}

// gradient wrt weights
__global__ void matmul_grad_w(
    const float* input,
    const float* dL_dout,
    float*       dW,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = d.din * d.dout;
    if (id >= total) return;
    unsigned int i = id / d.dout;
    unsigned int j = id % d.dout;
    float sum = 0.0f;
    for (unsigned int b = 0; b < d.batch; ++b) {
        sum += input[b * d.din + i] * dL_dout[b * d.dout + j];
    }
    dW[id] = sum / float(d.batch);
}

// relu backward mask
__global__ void relu_backward(
    const char* relu_mask,
    float*      dL_dout,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    if (relu_mask[id] == 0) dL_dout[id] = 0.0f;
}

// gradient wrt bias
__global__ void bias_grad_sum(
    const float* dL_dout,
    float*       dB,
    dims         d,
    unsigned int N
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.dout) return;
    float sum = 0.0f;
    for (unsigned int b = 0; b < d.batch; ++b) {
        sum += dL_dout[b * d.dout + j];
    }
    dB[j] = sum / float(d.batch);
}

// gradient wrt input
__global__ void matmul_grad_input(
    const float* dL_dout,
    const float* W,
    float*       dL_dx,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = d.batch * d.din;
    if (id >= total) return;
    unsigned int b = id / d.din;
    unsigned int i = id % d.din;
    float sum = 0.0f;
    for (unsigned int j = 0; j < d.dout; ++j) {
        sum += dL_dout[b * d.dout + j] * W[i * d.dout + j];
    }
    dL_dx[id] = sum;
}

// sgd weight update
__global__ void sgd_update_weight(
    float*       p,
    const float* g,
    float        lr,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = d.din * d.dout;
    if (id >= total) return;
    p[id] -= lr * g[id];
}

// sgd bias update
__global__ void sgd_update_bias(
    float*       p,
    const float* g,
    float        lr,
    dims         d,
    unsigned int N
) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= d.dout) return;
    p[id] -= lr * g[id];
}
