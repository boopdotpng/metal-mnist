#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <stdio.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static const uint32_t batch_size = 500;

struct Timing {
  std::string label;
  std::chrono::steady_clock::time_point start;
  Timing(const char *label)
      : label(label), start(std::chrono::steady_clock::now()) {}
  ~Timing() {
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%s: %f ms\n", label.c_str(), ms);
  }
};

struct kernel_launch {
  const char *name;
  MTLSize grid, block;
  std::vector<id<MTLBuffer>> buffers;
  id<MTLComputePipelineState> pipeline = nil;

  kernel_launch(const char *name) : name(name) {}

  kernel_launch &with_buffer(id<MTLBuffer> buf) {
    buffers.push_back(buf);
    return *this;
  }
  kernel_launch &with_buffers(std::initializer_list<id<MTLBuffer>> bufs) {
    buffers.insert(buffers.end(), bufs.begin(), bufs.end());
    return *this;
  }
  kernel_launch &with_grid(MTLSize g) {
    grid = g;
    return *this;
  }
  kernel_launch &with_block(MTLSize b) {
    block = b;
    return *this;
  }

  void encode(id<MTLComputeCommandEncoder> enc) const {
    [enc setComputePipelineState:pipeline];
    for (size_t i = 0; i < buffers.size(); ++i)
      [enc setBuffer:buffers[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
  }
};

struct metalcontext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  std::vector<kernel_launch> pending_kernels;

  std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
  std::unordered_map<std::string, id<MTLFunction>> fn_cache;

  void init() {
    device = MTLCreateSystemDefaultDevice();
    NSLog(@"using device %@", [device name]);
    queue = [device newCommandQueue];

    NSString *path = [[NSBundle mainBundle] pathForResource:@"fastkernels" ofType:@"metal"];
    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];

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

  id<MTLComputePipelineState> get_pipeline(const char *name) {
    auto it = pipeline_cache.find(name);
    if (it != pipeline_cache.end()) return it->second;

    id<MTLFunction> fn = nil;
    auto it_fn = fn_cache.find(name);
    if (it_fn != fn_cache.end())
      fn = it_fn->second;
    else 
      fn = [library newFunctionWithName:@(name)];

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
      k.pipeline = get_pipeline(k.name); k.encode(enc);
    }

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    pending_kernels.clear();
  }
}; metalcontext metal; // global instance

uint32_t read_big_endian_uint32(std::ifstream &ifs) {
  unsigned char bytes[4];
  if (!ifs.read((char *)bytes, 4))
    throw std::runtime_error("failed to read 4 bytes from file stream.");
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

// helper: allocate a shared buffer (for CPU readback or temporary use)
id<MTLBuffer> alloc(NSUInteger size) {
  return [metal.device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

struct MNIST {
  static constexpr uint32_t IMG_PIXELS = 28 * 28;
  static constexpr uint32_t TRAIN_CT = 60000;
  static constexpr uint32_t TEST_CT = 10000;

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
    size_t data_size = num_pixels * sizeof(unsigned char);

    id<MTLBuffer> staging = [metal.device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
    void *staging_ptr = [staging contents];
    if (!file.read((char *)staging_ptr, data_size))
      throw std::runtime_error("failed to read image data from: " + filename);

    id<MTLBuffer> dest_buf = is_train ? x_train : x_test;
    id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dest_buf destinationOffset:0 size:data_size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
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

    id<MTLBuffer> staging = [metal.device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
    void *staging_ptr = [staging contents];
    if (!file.read((char *)staging_ptr, data_size))
      throw std::runtime_error("failed to read label data from: " + filename);

    id<MTLBuffer> dest_buf = is_train ? y_train : y_test;
    id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dest_buf destinationOffset:0 size:data_size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    NSLog(@"loaded %u labels", num_labels);
  }

  void init() {
    x_train = [metal.device newBufferWithLength:sizeof(float)*TRAIN_CT*28*28 options:MTLResourceStorageModePrivate];
    y_train = [metal.device newBufferWithLength:sizeof(unsigned char)*TRAIN_CT options:MTLResourceStorageModePrivate];
    x_test = [metal.device newBufferWithLength:sizeof(float)*TEST_CT*28*28 options:MTLResourceStorageModePrivate];
    y_test = [metal.device newBufferWithLength:sizeof(unsigned char)*TEST_CT options:MTLResourceStorageModePrivate];

    load_images("./mnist/train-images.idx3-ubyte", true);
    load_labels("./mnist/train-labels.idx1-ubyte", true);
    load_images("./mnist/t10k-images.idx3-ubyte", false);
    load_labels("./mnist/t10k-labels.idx1-ubyte", false);
  }
};
MNIST dataset;

struct linear {
  id<MTLBuffer> weights, bias;
  uint in_dim, out_dim;
  bool use_relu;
  id<MTLBuffer> pre_act, post_act, dL_dout; // relu backward buffers

  linear(uint in_dim, uint out_dim, bool use_relu) : in_dim(in_dim), out_dim(out_dim), use_relu(use_relu) {}

  void init() {
    // TODO: zero init is fine for now, but use kaiming later
    weights = [metal.device newBufferWithLength:in_dim*out_dim*sizeof(float)  options:MTLResourceStorageModePrivate];
    bias = [metal.device newBufferWithLength:out_dim*sizeof(float)  options:MTLResourceStorageModePrivate];
  }

  id<MTLBuffer> operator()(id<MTLBuffer> x) {
    return forward(x);
  }

  id<MTLBuffer> forward(id<MTLBuffer> input) {
    pre_act = alloc(batch_size * out_dim * sizeof(float));
    post_act = use_relu ? alloc(batch_size * out_dim * sizeof(float)) : pre_act;

    kernel_launch("matmul_bias") 
      .with_buffers({input, weights, bias, pre_act})
      .with_grid(MTLSizeMake(batch_size * out_dim, 1, 1))
      .with_block(MTLSizeMake(128, 1, 1));

    if (use_relu) {
      kernel_launch("relu")
        .with_buffers({pre_act, post_act})
        .with_grid(...)
        .with_block(...);
    }

    return post_act;
  }
  void backward(id<MTLBuffer> input, id<MTLBuffer> dL_dout, id<MTLBuffer> &dL_din);
  void update(float lr);
  void relu() {
   kernel_launch("relu")
    .with_buffers()
    .with_grid()
    .with_block();
  }
};

struct model {
  linear l1, l2, l3; 
  id<MTLBuffer> relu1, relu2, logits, probs;

  model() : l1(784, 512, true), l2(512, 128, true), l3(128, 10, false) {}

  void init() { 
    l1.init(); l2.init(); l3.init();
   }

  id<MTLBuffer> forward(id<MTLBuffer> x) {
    id<MTLBuffer> x = l3(l2(l1(x)));
    metal.run_all();
  }

  void backward() { 

  }
};

void one_train() {}

int main(int argc, char *argv[]) {
  metal.init();
  dataset.init();
  model m;
  m.init();
  printf("help\n");
  return 0;
}
