#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>
#include <chrono>
#include <cmath> 
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

const uint32_t batch_size = 500;
constexpr MTLSize default_block{128,1,1};

struct dims { uint32_t batch, din, dout; };

struct Timing {
  std::string label;
  std::chrono::steady_clock::time_point start;

  Timing(const char *label) : label(label) {}

  void begin() {
    start = std::chrono::steady_clock::now();
  }

  void end() {
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - start).count();
    printf("%s: %.3f ms\n", label.c_str(), ms);
  }
};

struct launch {
  const char *name;
  MTLSize grid, block;
  std::vector<id<MTLBuffer>> buffers;
  id<MTLComputePipelineState> pipeline = nil;

  launch(const char *name) : name(name) {}

  launch &with_buffers(std::initializer_list<id<MTLBuffer>> bufs) {
    buffers.insert(buffers.end(), bufs.begin(), bufs.end()); return *this;
  }

  launch &with_grid(MTLSize g) {
    grid = g; return *this;
  }

  launch &with_block(MTLSize b) {
    block = b; return *this;
  }

  void encode(id<MTLComputeCommandEncoder> enc) const {
    [enc setComputePipelineState:pipeline];
    for (size_t i = 0; i < buffers.size(); ++i)
      [enc setBuffer:buffers[i] offset:0 atIndex:i];

    [enc dispatchThreads:grid threadsPerThreadgroup:block];
  }
};

struct metalcontext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  std::vector<launch> pending_kernels;

  std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
  std::unordered_map<std::string, id<MTLFunction>> fn_cache;

  void init() {
    device = MTLCreateSystemDefaultDevice();
    NSLog(@"using device %@", [device name]);
    queue = [device newCommandQueue];

    NSString *src = [NSString stringWithContentsOfFile:@"./fast/kernels.metal" encoding:NSUTF8StringEncoding error:nil];

    NSError *err = nil;
    library = [device newLibraryWithSource:src options:nil error:&err];
    if (!library) {
      NSLog(@"metal compile error: %@", err);
      exit(1);
    }
    NSLog(@"metal ok");
  }

  id<MTLBuffer> alloc(uint size) {
    return [device newBufferWithLength:size options:MTLResourceStorageModeShared];
  }
  id<MTLBuffer> alloc_gpu(uint size) {
    return [device newBufferWithLength:size options:MTLResourceStorageModePrivate];
  }

  void enqueue(const char *name, std::initializer_list<id<MTLBuffer>> bufs, size_t grid_x) {
    pending_kernels.push_back(launch(name));
    pending_kernels.back()
      .with_buffers(bufs)
      .with_grid({grid_x, 1, 1})
      .with_block(default_block);
  }

  id<MTLComputePipelineState> get_pipeline(const char *name) {
    auto it = pipeline_cache.find(name);
    if (it != pipeline_cache.end()) return it->second;

    id<MTLFunction> fn = [library newFunctionWithName:@(name)];
    NSError *err = nil;
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];

    pipeline_cache[name] = p;
    fn_cache[name] = fn;
    return p;
  }

  void run_all() {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    for (auto &k : pending_kernels) {
      k.pipeline = get_pipeline(k.name);
      k.encode(enc);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    pending_kernels.clear();
  }

  id<MTLBuffer> to_cpu(id<MTLBuffer> buf) {
    id<MTLBuffer> shared = alloc(buf.length);
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:buf sourceOffset:0 toBuffer:shared destinationOffset:0 size:buf.length];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return shared;
  }
}; metalcontext metal; 

uint32_t read_big_endian_uint32(std::ifstream &ifs) {
  unsigned char bytes[4];
  if (!ifs.read((char *)bytes, 4))
    throw std::runtime_error("failed to read 4 bytes from file stream.");
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

struct MNIST {
  static constexpr uint32_t IMG_PIXELS = 28 * 28, TRAIN_CT = 60000, TEST_CT = 10000;
  id<MTLBuffer> x_train, y_train, x_test, y_test;

  void load_images(const std::string &filename, bool is_train) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("failed to open image file: " + filename);

    uint32_t magic = read_big_endian_uint32(file);
    if (magic != 0x00000803)
      throw std::runtime_error("invalid magic number in image file: " + filename);

    uint32_t num_images = read_big_endian_uint32(file);
    uint32_t rows = read_big_endian_uint32(file);
    uint32_t cols = read_big_endian_uint32(file);
    size_t num_pixels = num_images * rows * cols;

    id<MTLBuffer> staging = metal.alloc(num_pixels);
    void *staging_ptr = staging.contents;
    if (!file.read((char *)staging_ptr, num_pixels))
      throw std::runtime_error("failed to read image data from: " + filename);

    metal.enqueue("normalize_image", {staging, is_train ? x_train : x_test}, num_images * IMG_PIXELS);
    metal.run_all(); // staging goes out of scope
    NSLog(@"loaded %u images", num_images);
  }

  void load_labels(const std::string &filename, bool is_train) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("failed to open label file: " + filename);

    uint32_t magic = read_big_endian_uint32(file);
    if (magic != 0x00000801)
      throw std::runtime_error("invalid magic number in label file: " + filename);

    uint32_t num_labels = read_big_endian_uint32(file);
    size_t data_size = num_labels * sizeof(unsigned char);

    id<MTLBuffer> staging = metal.alloc(data_size); 
    void *staging_ptr = staging.contents; 
    if (!file.read((char *)staging_ptr, data_size))
      throw std::runtime_error("failed to read label data from: " + filename);

    id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:is_train ? y_train : y_test destinationOffset:0 size:data_size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    NSLog(@"loaded %u labels", num_labels);
  }

  void init() {
    x_train = metal.alloc_gpu(sizeof(float)*TRAIN_CT*28*28);
    y_train = metal.alloc_gpu(sizeof(unsigned char)*TRAIN_CT);
    x_test = metal.alloc_gpu(sizeof(float)*TEST_CT*28*28);
    y_test = metal.alloc_gpu(sizeof(unsigned char)*TEST_CT);

    load_images("./mnist/train-images.idx3-ubyte", true);
    load_labels("./mnist/train-labels.idx1-ubyte", true);
    load_images("./mnist/t10k-images.idx3-ubyte", false);
    load_labels("./mnist/t10k-labels.idx1-ubyte", false);
  }
}; MNIST dataset;

struct linear {
  uint in_dim, out_dim;
  id<MTLBuffer> w, b, dW, dB, input, relu_mask, output, lr_buf, dL_dx, dims_buf;
  bool use_relu;

  linear(uint in_dim, uint out_dim, bool use_relu)
    : in_dim(in_dim), out_dim(out_dim), use_relu(use_relu) {}

  void init() {
    w = metal.alloc_gpu(in_dim*out_dim*sizeof(float));
    b = metal.alloc_gpu(out_dim*sizeof(float));
    dW = metal.alloc_gpu(in_dim * out_dim * sizeof(float));
    dB = metal.alloc_gpu(out_dim * sizeof(float));
    dL_dx = metal.alloc_gpu(batch_size * in_dim * sizeof(float));
    output = metal.alloc_gpu(batch_size * out_dim * sizeof(float));
    dims_buf = metal.alloc(sizeof(dims));
    dims params = { batch_size, in_dim, out_dim };
    memcpy(dims_buf.contents, &params, sizeof(dims));

    lr_buf = metal.alloc(sizeof(float));
    if (use_relu) relu_mask = metal.alloc_gpu(batch_size * out_dim * sizeof(char));

    layer_init();
  }

  void layer_init() {
    id<MTLBuffer> staging_w = metal.alloc(in_dim * out_dim * sizeof(float)); 
    id<MTLBuffer> staging_b = metal.alloc(out_dim * sizeof(float));         
    float* w_ptr = (float*)staging_w.contents;
    float* b_ptr = (float*)staging_b.contents;

    std::mt19937 gen(1337);
    // std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count()); 

    float kaiming_bound_w = sqrtf(6.0f / in_dim);
    std::uniform_real_distribution<float> dist_w(-kaiming_bound_w, kaiming_bound_w);
    for (uint i = 0; i < in_dim * out_dim; ++i) {
        w_ptr[i] = dist_w(gen);
    }

    float bias_bound = 1.0f / sqrtf(in_dim);
    std::uniform_real_distribution<float> dist_b(-bias_bound, bias_bound);
    for (uint i = 0; i < out_dim; ++i) {
        b_ptr[i] = 0.0f; 
    }

    id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging_w sourceOffset:0 toBuffer:w destinationOffset:0 size:staging_w.length];
    [blit copyFromBuffer:staging_b sourceOffset:0 toBuffer:b destinationOffset:0 size:staging_b.length];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
  }

  id<MTLBuffer> forward(id<MTLBuffer> input) {
    this->input= input;

    if (use_relu) {
      metal.enqueue("matmul_bias_relu", {input, w, b, relu_mask, output, dims_buf}, batch_size * out_dim);
    } else {
      metal.enqueue("matmul_bias", {input, w, b, output, dims_buf}, batch_size * out_dim);
    }
    return output;
  }

  id<MTLBuffer> backward(id<MTLBuffer> dL_dout) {
    if (use_relu) 
      metal.enqueue("relu_backward", {relu_mask, dL_dout}, batch_size * out_dim);

    metal.enqueue("matmul_grad_w", {input, dL_dout, dW, dims_buf}, in_dim * out_dim);

    metal.enqueue("bias_grad_sum", {dL_dout, dB, dims_buf}, out_dim);
    metal.enqueue("matmul_grad_input", {dL_dout, w, dL_dx, dims_buf}, batch_size * in_dim);
    return dL_dx;
  }

  void update(float lr) {
    *(float*)lr_buf.contents = lr;
    metal.enqueue("sgd_update_weight", {w, dW, lr_buf, dims_buf}, in_dim * out_dim);
    metal.enqueue("sgd_update_bias", {b, dB, lr_buf, dims_buf}, out_dim);
  }

  void zero_grad() {
    metal.enqueue("zero_buffer", {dW}, in_dim*out_dim);
    metal.enqueue("zero_buffer", {dB}, out_dim);
  }
};

struct CrossEntropyLoss {
  linear &lin;
  // dL_dlogits is the backprop that chains into the linear layer
  id<MTLBuffer> logits, loss_buf, dL_dlogits, dims_buf;

  CrossEntropyLoss(linear &lin) : lin(lin) {}

  void init() {
    loss_buf = metal.alloc(batch_size * sizeof(float));
    dL_dlogits = metal.alloc_gpu(batch_size * lin.out_dim * sizeof(float));
    logits = lin.output;
    dims_buf = metal.alloc(sizeof(dims));
    dims params = { batch_size, lin.in_dim, lin.out_dim };
    memcpy(dims_buf.contents, &params, sizeof(dims));
  }

  void forward(id<MTLBuffer> labels) { 
    metal.enqueue("cross_entropy", {logits, labels, loss_buf, dims_buf}, batch_size);
  }

  id<MTLBuffer> backward(id<MTLBuffer> labels) {
    metal.enqueue("cross_entropy_backward", {logits, labels, dL_dlogits, dims_buf}, batch_size * lin.out_dim);
    return dL_dlogits;
  }

  id<MTLBuffer> loss() { return loss_buf; }
};

struct model {
  linear l1, l2, l3;
  CrossEntropyLoss loss;

  model() : l1(784, 512, true), l2(512, 128, true), l3(128, 10, false), loss(l3) {}

  void init() { 
    l1.init(); l2.init(); l3.init(); loss.init();
  }

  id<MTLBuffer> run(id<MTLBuffer> x, id<MTLBuffer> labels, float lr) {
    l1.zero_grad();
    l2.zero_grad();
    l3.zero_grad();

    // forward pass
    id<MTLBuffer> out1 = l1.forward(x);
    id<MTLBuffer> out2 = l2.forward(out1);
    id<MTLBuffer> logits = l3.forward(out2);
    loss.forward(labels);

    // backward pass
    id<MTLBuffer> dL_dlogits = loss.backward(labels);
    id<MTLBuffer> dL2 = l3.backward(dL_dlogits);
    id<MTLBuffer> dL1 = l2.backward(dL2);
    id<MTLBuffer> dL0 = l1.backward(dL1);
    // working
    
    // update
    l1.update(lr);
    l2.update(lr);
    l3.update(lr);

    metal.run_all(); 
    return loss.loss();
  }
};

struct Batch {
  id<MTLBuffer> x;
  id<MTLBuffer> y;
};

Batch get_batch(uint batch) {
  Batch b;
  size_t x_offset = batch * batch_size * 28 * 28 * sizeof(float);
  size_t x_size = batch_size * 28 * 28 * sizeof(float);
  b.x = metal.alloc(x_size);

  size_t y_offset = batch * batch_size * sizeof(unsigned char);
  size_t y_size = batch_size * sizeof(unsigned char);
  b.y = metal.alloc(y_size);

  id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  [blit copyFromBuffer:dataset.x_train sourceOffset:x_offset
               toBuffer:b.x destinationOffset:0 size:x_size];
  [blit copyFromBuffer:dataset.y_train sourceOffset:y_offset
               toBuffer:b.y destinationOffset:0 size:y_size];
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];

  return b;
}

void run_one(model &m, MNIST &dataset, float lr) {
    float sum = 0.0f;
    uint total_batches = MNIST::TRAIN_CT / batch_size;
    uint total_samples = total_batches * batch_size;

    Timing timer("epoch");
    timer.begin();
    for (uint32_t batch = 0; batch < total_batches; ++batch) {
        Batch b = get_batch(batch);
        id<MTLBuffer> loss_buf = m.run(b.x, b.y, lr);
        float *loss_cpu = (float*)loss_buf.contents;
        for (int i = 0; i < batch_size; ++i)
            sum += loss_cpu[i];
    }
    timer.end();
    printf("epoch loss: %f\n", sum / total_samples);
}

void train(model &m, MNIST &dataset, uint epoch) {
  // room for optimizer with this setup
  const float lr = 0.01f;
  for (uint8_t i = 0; i < epoch; i++) run_one(m, dataset, lr);
}

int main(int argc, char *argv[]) {
  metal.init();
  dataset.init();
  model m;
  m.init();
  train(m, dataset, 10);
  return 0;
}
