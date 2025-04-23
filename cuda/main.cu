#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <chrono>

const uint32_t batch_size = 500;
cudaStream_t stream;

struct dims { uint32_t batch, din, dout; };

struct Timing {
  std::string label;
  std::chrono::steady_clock::time_point start;
  void begin() { start = std::chrono::steady_clock::now(); }
  double end() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
  }
};

#include "kernels.cu"

inline int calc_blocks(int total, int per_block = 256) {
  return (total + per_block - 1) / per_block;
}

#define launch(kernel, grid, block, ...) \
  kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)

template<typename T>
T* gpu_alloc(size_t count) {
  T* ptr = nullptr;
  cudaMalloc(&ptr, count * sizeof(T));  // error handling removed
  return ptr;
}


struct MNIST {
  static constexpr unsigned IMG_PIXELS = 28u * 28u;
  static constexpr unsigned TRAIN_CT    = 60000u;
  static constexpr unsigned TEST_CT     = 10000u;

  float*          x_train = nullptr;
  unsigned char*  y_train = nullptr;
  float*          x_test  = nullptr;
  unsigned char*  y_test  = nullptr;

  static uint32_t read_u32(std::ifstream& f) {
    uint32_t v; f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return __builtin_bswap32(v);
  }

  template<typename T>
  void load_data(const std::string& path,
                 std::vector<T>&      host,
                 T*                   device,
                 bool                 is_image)
  {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open " + path);

    uint32_t magic = read_u32(f), count = read_u32(f);
    size_t elems = is_image ? size_t(count) * IMG_PIXELS : size_t(count);
    host.resize(elems);

    if (is_image) {
      uint32_t rows = read_u32(f), cols = read_u32(f);
      std::vector<unsigned char> raw(elems);
      f.read(reinterpret_cast<char*>(raw.data()), raw.size());
      for (size_t i = 0; i < elems; ++i) host[i] = raw[i] / 255.0f;
    } else {
      f.read(reinterpret_cast<char*>(host.data()), elems);
    }

    cudaMemcpy(device, host.data(), elems * sizeof(T), cudaMemcpyHostToDevice);
    std::printf("loaded %zu %s from %s\n",
                elems,
                is_image ? "pixels" : "labels",
                path.c_str());
  }

  void init() {
    cudaSetDevice(0);
    cudaStreamCreate(&stream);

    x_train = gpu_alloc<float>(TRAIN_CT * IMG_PIXELS);
    y_train = gpu_alloc<unsigned char>(TRAIN_CT);
    x_test  = gpu_alloc<float>(TEST_CT * IMG_PIXELS);
    y_test  = gpu_alloc<unsigned char>(TEST_CT);

    std::vector<float>        img_buf;
    std::vector<unsigned char> lbl_buf;

    load_data("./mnist/train-images.idx3-ubyte", img_buf, x_train, true);
    load_data("./mnist/train-labels.idx1-ubyte", lbl_buf, y_train, false);
    load_data("./mnist/t10k-images.idx3-ubyte",  img_buf, x_test,  true);
    load_data("./mnist/t10k-labels.idx1-ubyte",  lbl_buf, y_test,  false);
  }
};

struct linear {
  uint32_t in_dim, out_dim;
  float *w, *b, *dW, *dB, *input, *output, *dL_dx;
  char  *relu_mask = nullptr;
  bool   use_relu;
  dims   params;

  linear(uint32_t din, uint32_t dout, bool relu)
    : in_dim(din), out_dim(dout),
      use_relu(relu),
      params{batch_size, din, dout}
  {}

  void init() {
    size_t Wsz = size_t(in_dim) * out_dim;
    w       = gpu_alloc<float>(Wsz);
    dW      = gpu_alloc<float>(Wsz);
    b       = gpu_alloc<float>(out_dim);
    dB      = gpu_alloc<float>(out_dim);
    output  = gpu_alloc<float>(batch_size * out_dim);
    dL_dx   = gpu_alloc<float>(batch_size * in_dim);
    if (use_relu)
      relu_mask = gpu_alloc<char>(batch_size * out_dim);

    std::vector<float> hw(Wsz), hb(out_dim, 0.0f);
    std::mt19937       gen(1337);
    float bound = std::sqrt(6.0f / in_dim);
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (auto& v : hw) v = dist(gen);

    cudaMemcpy(w, hw.data(), Wsz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb.data(), out_dim*sizeof(float), cudaMemcpyHostToDevice);
  }

  float* forward() {
    int total = batch_size * out_dim;
    if (use_relu) {
      launch(matmul_bias_relu,
             calc_blocks(total), 256,
             input, w, b, relu_mask, output, params, total);
    } else {
      launch(matmul_bias,
             calc_blocks(total), 256,
             input, w, b, output, params, total);
    }
    return output;
  }

  float* backward(float* dLout) {
    int totalO = batch_size * out_dim;
    if (use_relu) {
      launch(relu_backward,
             calc_blocks(totalO), 256,
             relu_mask, dLout, totalO);
    }
    launch(matmul_grad_w,
           calc_blocks(in_dim * out_dim), 256,
           input, dLout, dW, params, in_dim*out_dim);
    launch(bias_grad_sum,
           calc_blocks(out_dim), 256,
           dLout, dB, params, out_dim);
    launch(matmul_grad_input,
           calc_blocks(batch_size * in_dim), 256,
           dLout, w, dL_dx, params, batch_size*in_dim);

    return dL_dx;
  }

  void update(float lr) {
    launch(sgd_update_weight,
           calc_blocks(in_dim*out_dim), 256,
           w, dW, lr, params, in_dim*out_dim);
    launch(sgd_update_bias,
           calc_blocks(out_dim), 256,
           b, dB, lr, params, out_dim);
  }

  void zero_grad() {
    cudaMemsetAsync(dW, 0, in_dim*out_dim*sizeof(float), stream);
    cudaMemsetAsync(dB, 0, out_dim*sizeof(float), stream);
  }
};

struct CrossEntropyLoss {
  linear&          lin;
  float           *logits, *loss_buf, *dL_dlogits;
  unsigned char   *labels;
  dims             params;

  CrossEntropyLoss(linear& L)
    : lin(L), logits(nullptr), labels(nullptr),
      loss_buf(nullptr), dL_dlogits(nullptr),
      params{batch_size, L.in_dim, L.out_dim}
  {}

  void init() {
    loss_buf   = gpu_alloc<float>(batch_size);
    dL_dlogits = gpu_alloc<float>(batch_size * lin.out_dim);
  }

  void forward() {
    launch(cross_entropy,
           calc_blocks(batch_size), 256,
           logits, labels, loss_buf, params, batch_size);
  }

  float* backward() {
    launch(cross_entropy_backward,
           calc_blocks(batch_size * lin.out_dim), 256,
           logits, labels, dL_dlogits, params, batch_size*lin.out_dim);
    return dL_dlogits;
  }

  float* loss() const { return loss_buf; }
};

struct model {
  linear            l1, l2, l3;
  CrossEntropyLoss  loss_module;

  model()
    : l1(784, 512, true),
      l2(512, 128, true),
      l3(128,  10, false),
      loss_module(l3)
  {}

  void init() {
    l1.init(); l2.init(); l3.init();
    loss_module.init();
  }

  void set_batch(int bidx, MNIST& data) {
    size_t x_offset = size_t(bidx) * batch_size * MNIST::IMG_PIXELS;
    size_t y_offset = size_t(bidx) * batch_size;

    l1.input        = data.x_train + x_offset;
    l2.input        = l1.forward(); 
    l3.input        = l2.forward(); 
    loss_module.logits = l3.forward();
    loss_module.labels = data.y_train + y_offset;
  }

  float* run(int bidx, float lr, MNIST& data) {
    set_batch(bidx, data);
    l1.zero_grad(); l2.zero_grad(); l3.zero_grad();

    loss_module.forward();
    float* d1 = loss_module.backward();
    float* d2 = l3.backward(d1);
    float* d3 = l2.backward(d2);
    (void)l1.backward(d3);

    l1.update(lr);
    l2.update(lr);
    l3.update(lr);

    cudaStreamSynchronize(stream);
    return loss_module.loss();
  }
};

void run_one(model& m, MNIST& data, float lr) {
  double sum = 0.0;
  int total_batches = MNIST::TRAIN_CT / batch_size;
  std::vector<float> host_loss(batch_size);

  Timing t; 
  t.begin();
  for (int b = 0; b < total_batches; ++b) {
    float* dev_loss = m.run(b, lr, data);
    cudaMemcpyAsync(host_loss.data(), dev_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (float v : host_loss) sum += v;
  }
  std::printf("epoch loss: %f, took %fms\n", sum / (total_batches * batch_size), t.end());
}

int main() {
  MNIST dataset;
  dataset.init();

  model m;
  m.init();

  const float lr = 0.01f;
  for (int e = 0; e < 10; ++e) {
    run_one(m, dataset, lr);
  }
  cudaDeviceSynchronize();
  return 0;
}
