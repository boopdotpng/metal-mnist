#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <chrono>

using namespace std;

void launch_kernel(const char *name, std::initializer_list<id<MTLBuffer>> buffers, MTLSize grid, MTLSize block);

struct MatmulParams { uint32_t M, K, N; };
struct InitParams {uint32_t param_count, fan_in, fan_out; };

// kernel compilation cache 
static std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;

struct Timer {
  chrono::high_resolution_clock::time_point start; 
  Timer() { reset(); }
  void reset() {
    start = chrono::high_resolution_clock::now();
  }
  double elapsed_ms() const {
    auto now = chrono::high_resolution_clock::now();
    return chrono::duration<double, std::milli>(now-start).count();
  }
};

struct metalcontext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  void init() {
    device = MTLCreateSystemDefaultDevice();
    NSLog(@"using device %@", [device name]);
    queue = [device newCommandQueue];
    
    NSError *err = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"kernels" ofType:@"metal"];
    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!src) {
        NSLog(@"failed to read metal source: %@", err);
        throw runtime_error("metal source read failed");
    }

    library = [device newLibraryWithSource:src options:nil error:&err];
    if (!library) {
        NSLog(@"metal compile error: %@", err);
        throw runtime_error("metal library compile failed");
    }
  }
};
static metalcontext metal;

struct mnist_data {
  // final usable float32 data
  id<MTLBuffer> xtrain; 
  id<MTLBuffer> ytrain;
  id<MTLBuffer> xtest;
  id<MTLBuffer> ytest;

  // raw uint8 input buffers
  id<MTLBuffer> xtrain_un;
  id<MTLBuffer> xtest_un;

  void init(id<MTLDevice> device) {
    xtrain_un = [device newBufferWithLength:60000 * 28 * 28 options:MTLResourceStorageModeShared];
    xtest_un  = [device newBufferWithLength:10000 * 28 * 28 options:MTLResourceStorageModeShared];

    xtrain = [device newBufferWithLength:60000 * 28 * 28 * sizeof(float) options:MTLResourceStorageModeShared];
    xtest  = [device newBufferWithLength:10000 * 28 * 28 * sizeof(float) options:MTLResourceStorageModeShared];

    ytrain = [device newBufferWithLength:60000 options:MTLResourceStorageModeShared];
    ytest  = [device newBufferWithLength:10000 options:MTLResourceStorageModeShared];
  }

  void normalize() {
    launch_kernel("normalize_image", {xtrain_un, xtrain}, MTLSizeMake(60000 * 28 * 28, 1, 1), MTLSizeMake(128, 1, 1));
    launch_kernel("normalize_image", {xtest_un, xtest}, MTLSizeMake(10000 * 28 * 28, 1, 1), MTLSizeMake(128, 1, 1));
    xtrain_un = nil;
    xtest_un = nil;
  }
};
static mnist_data mnist;

void launch_kernel(const char *name, std::initializer_list<id<MTLBuffer>> buffers, MTLSize grid, MTLSize block) {
  NSError *err = nil;

  id<MTLComputePipelineState> pipeline; 
  auto it = pipeline_cache.find(name);
  if (it != pipeline_cache.end()) {
    pipeline = it->second;
  } else {
    id<MTLFunction> fn = [metal.library newFunctionWithName:@(name)];
    if (!fn) throw runtime_error("kernel not found");
    NSError *err = nil;
    pipeline = [metal.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) throw runtime_error("pipeline failed");
    pipeline_cache[name] = pipeline;
  }

  id<MTLCommandBuffer> cmd = [metal.queue commandBuffer]; 
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];

  int i = 0; 
  for(id<MTLBuffer> buf: buffers) {
    [enc setBuffer: buf offset:0 atIndex:i++];
  }

  [enc dispatchThreads:grid threadsPerThreadgroup:block];

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
}

// class model
struct model {
  // linear 1 
  // only one float32 layer now for testing.
  id<MTLBuffer> linear1; 
  id<MTLBuffer> bias1; 
  id<MTLBuffer> linear2; 
  id<MTLBuffer> bias2; 
  id<MTLBuffer> linear3;
  id<MTLBuffer> bias3; 

  void init() {
    InitParams p = {784*512, 784, 512};
    id<MTLBuffer> param_buf = [metal.device newBufferWithLength:sizeof(InitParams) options:MTLResourceStorageModeShared];
    memcpy(param_buf.contents, &p, sizeof(p));

    linear1 = [metal.device newBufferWithLength:28*28*512 * sizeof(float) options:MTLResourceStorageModeShared];
    bias1 = [metal.device newBufferWithLength:512 * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear1, param_buf}, MTLSizeMake(28*28*512,1,1), MTLSizeMake(128,1,1));


    p = {512*128, 512, 128};
    memcpy(param_buf.contents, &p, sizeof(p));
    linear2 = [metal.device newBufferWithLength:512*128 * sizeof(float) options:MTLResourceStorageModeShared];
    bias2 = [metal.device newBufferWithLength:128 * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear2, param_buf}, MTLSizeMake(512*128,1,1), MTLSizeMake(128,1,1));

    p = {128*10, 128, 10};
    memcpy(param_buf.contents, &p, sizeof(p));
    linear3 = [metal.device newBufferWithLength:128*10 * sizeof(float) options:MTLResourceStorageModeShared];
    bias3 = [metal.device newBufferWithLength:10 * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear3, param_buf}, MTLSizeMake(128*10,1,1), MTLSizeMake(128,1,1));

    // bias1 init
    p = {512, 512, 512}; // param_count, fan_in, fan_out
    memcpy(param_buf.contents, &p, sizeof(p));
    launch_kernel("init_random_weights", {bias1, param_buf}, MTLSizeMake(512, 1, 1), MTLSizeMake(128, 1, 1));

    // bias2 init
    p = {128, 128, 128};
    memcpy(param_buf.contents, &p, sizeof(p));
    launch_kernel("init_random_weights", {bias2, param_buf}, MTLSizeMake(128, 1, 1), MTLSizeMake(128, 1, 1));

    // bias3 init
    p = {10, 10, 10};
    memcpy(param_buf.contents, &p, sizeof(p));
    launch_kernel("init_random_weights", {bias3, param_buf}, MTLSizeMake(10, 1, 1), MTLSizeMake(10, 1, 1)); 
  }
};
static model mnist_model;

uint32_t read_big_endian_uint32(ifstream &ifs) {
  unsigned char bytes[4];
  ifs.read((char*)bytes, 4);
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8)  | uint32_t(bytes[3]);
}

void load_images(const string &filename, id<MTLBuffer> buffer, uint32_t &num_images, uint32_t &rows, uint32_t &cols) {
  ifstream file(filename, ios::binary);
  if (!file) throw runtime_error("error opening " + filename);

  uint32_t magic = read_big_endian_uint32(file);
  num_images = read_big_endian_uint32(file);
  rows = read_big_endian_uint32(file);
  cols = read_big_endian_uint32(file);

  void *dest = buffer.contents;
  file.read((char*)dest, num_images * rows * cols);
}

void load_labels(const string &filename, id<MTLBuffer> buffer, uint32_t &num_labels) {
  ifstream file(filename, ios::binary);
  if (!file) throw runtime_error("error opening " + filename);

  uint32_t magic = read_big_endian_uint32(file);
  num_labels = read_big_endian_uint32(file);

  void *dest = buffer.contents;
  file.read((char*)dest, num_labels);
}

float train(uint32_t batch_index) {
  const int batch_size = 200;
  const int input_dim = 784;
  const int hidden1 = 512;
  const int hidden2 = 128;
  const int output_dim = 10;
  float lr = 0.01f;

  // buffers
  id<MTLBuffer> shape_buf = [metal.device newBufferWithLength:sizeof(MatmulParams) options:MTLResourceStorageModeShared];
  id<MTLBuffer> batch = [metal.device newBufferWithLength:batch_size * input_dim * sizeof(float) options:MTLResourceStorageModeShared];
  // note: labels are stored as int; ideally use uchar but we'll leave as int here
  id<MTLBuffer> batch_labels = [metal.device newBufferWithLength:batch_size * sizeof(unsigned char) options:MTLResourceStorageModeShared];
  id<MTLBuffer> offset_buf = [metal.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> bias_dim = [metal.device newBufferWithLength:sizeof(uint) options:MTLResourceStorageModeShared];
  id<MTLBuffer> losses_buf = [metal.device newBufferWithLength:batch_size * sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> lr_buf = [metal.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
  memcpy(lr_buf.contents, &lr, sizeof(float));

  uint32_t offset = batch_index * batch_size;
  memcpy([offset_buf contents], &offset, sizeof(uint32_t));

  // copy batch and labels from data
  launch_kernel("copy_batch", {mnist.xtrain, batch, offset_buf},
                MTLSizeMake(batch_size * input_dim, 1, 1), MTLSizeMake(128, 1, 1));
  launch_kernel("copy_batch_labels", {mnist.ytrain, batch_labels, offset_buf},
                MTLSizeMake(batch_size, 1, 1), MTLSizeMake(128, 1, 1));

  // ------------------
  // FORWARD PASS
  // ------------------

  // layer 1: [B x 784] * [784 x 512] = [B x 512]
  id<MTLBuffer> buf1 = [metal.device newBufferWithLength:batch_size * hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape1 = {batch_size, input_dim, hidden1};
  memcpy([shape_buf contents], &shape1, sizeof(shape1));
  launch_kernel("matmul", {batch, mnist_model.linear1, buf1, shape_buf},
                MTLSizeMake(batch_size * hidden1, 1, 1), MTLSizeMake(128, 1, 1));
  uint d = hidden1;
  memcpy(bias_dim.contents, &d, sizeof(uint));
  launch_kernel("add_bias", {buf1, mnist_model.bias1, bias_dim},
                MTLSizeMake(batch_size * d, 1, 1), MTLSizeMake(64, 1, 1));
  launch_kernel("relu", {buf1},
                MTLSizeMake(batch_size * hidden1, 1, 1), MTLSizeMake(128, 1, 1));

  // layer 2: [B x 512] * [512 x 128] = [B x 128]
  id<MTLBuffer> buf2 = [metal.device newBufferWithLength:batch_size * hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape2 = {batch_size, hidden1, hidden2};
  memcpy([shape_buf contents], &shape2, sizeof(shape2));
  launch_kernel("matmul", {buf1, mnist_model.linear2, buf2, shape_buf},
                MTLSizeMake(batch_size * hidden2, 1, 1), MTLSizeMake(128, 1, 1));
  d = hidden2;
  memcpy(bias_dim.contents, &d, sizeof(uint));
  launch_kernel("add_bias", {buf2, mnist_model.bias2, bias_dim},
                MTLSizeMake(batch_size * d, 1, 1), MTLSizeMake(64, 1, 1));
  // cache pre-relu buf2 for backward; then apply relu
  launch_kernel("relu", {buf2},
                MTLSizeMake(batch_size * hidden2, 1, 1), MTLSizeMake(128, 1, 1));

  // layer 3: [B x 128] * [128 x 10] = [B x 10]
  id<MTLBuffer> buf3 = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape3 = {batch_size, hidden2, output_dim};
  memcpy([shape_buf contents], &shape3, sizeof(shape3));
  launch_kernel("matmul", {buf2, mnist_model.linear3, buf3, shape_buf},
                MTLSizeMake(batch_size * output_dim, 1, 1), MTLSizeMake(128, 1, 1));
  d = output_dim;
  memcpy(bias_dim.contents, &d, sizeof(uint));
  launch_kernel("add_bias", {buf3, mnist_model.bias3, bias_dim},
                MTLSizeMake(batch_size * d, 1, 1), MTLSizeMake(64, 1, 1));
  id<MTLBuffer> probs = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float) options:MTLResourceStorageModeShared];
  launch_kernel("softmax", {buf3, probs, bias_dim},
              MTLSizeMake(batch_size * d, 1, 1), MTLSizeMake(64, 1, 1));

  // compute loss BEFORE backward pass
  launch_kernel("cross_entropy_loss", {probs, batch_labels, losses_buf, bias_dim},
                MTLSizeMake(batch_size, 1, 1), MTLSizeMake(128, 1, 1));
  float *losses_ptr = (float *)losses_buf.contents;
  float loss_sum = 0;
  for (int i = 0; i < batch_size; ++i) {
    loss_sum += losses_ptr[i];
  }
  float loss = loss_sum / batch_size;

  // ------------------
  // BACKWARD PASS
  // ------------------
  
  // allocate gradient buffer for final layer (dL/dlogits)
  id<MTLBuffer> dL_dlogits = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
  launch_kernel("softmax_cross_entropy_backward", {probs, batch_labels, dL_dlogits, bias_dim},
                MTLSizeMake(batch_size * output_dim, 1, 1), MTLSizeMake(64, 1, 1));
  
  // layer 3 backward:
  // gradW3 = (buf2)ᵀ @ dL_dlogits, and dA2 = dL_dlogits @ (linear3)ᵀ
  memcpy([shape_buf contents], &shape3, sizeof(shape3));
  id<MTLBuffer> gradW3 = [metal.device newBufferWithLength:hidden2 * output_dim * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {buf2, dL_dlogits, gradW3, shape_buf},
                MTLSizeMake(hidden2 * output_dim, 1, 1), MTLSizeMake(64, 1, 1));
  
  id<MTLBuffer> dA2 = [metal.device newBufferWithLength:batch_size * hidden2 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_input", {dL_dlogits, mnist_model.linear3, dA2, shape_buf},
                MTLSizeMake(batch_size * hidden2, 1, 1), MTLSizeMake(64, 1, 1));
  
  // relu backward for layer 2
  launch_kernel("relu_backward", {buf2, dA2},
                MTLSizeMake(batch_size * hidden2, 1, 1), MTLSizeMake(128, 1, 1));
  
  // layer 2 backward:
  MatmulParams shape2_bwd = {batch_size, hidden1, hidden2};
  memcpy([shape_buf contents], &shape2_bwd, sizeof(shape2_bwd));
  id<MTLBuffer> gradW2 = [metal.device newBufferWithLength:hidden1 * hidden2 * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {buf1, dA2, gradW2, shape_buf},
                MTLSizeMake(hidden1 * hidden2, 1, 1), MTLSizeMake(64, 1, 1));
  
  id<MTLBuffer> dA1 = [metal.device newBufferWithLength:batch_size * hidden1 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_input", {dA2, mnist_model.linear2, dA1, shape_buf},
                MTLSizeMake(batch_size * hidden1, 1, 1), MTLSizeMake(64, 1, 1));
  
  // relu backward for layer 1
  launch_kernel("relu_backward", {buf1, dA1},
                MTLSizeMake(batch_size * hidden1, 1, 1), MTLSizeMake(128, 1, 1));
  
  // layer 1 backward:
  MatmulParams shape1_bwd = {batch_size, input_dim, hidden1};
  memcpy([shape_buf contents], &shape1_bwd, sizeof(shape1_bwd));
  id<MTLBuffer> gradW1 = [metal.device newBufferWithLength:input_dim * hidden1 * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {batch, dA1, gradW1, shape_buf},
                MTLSizeMake(input_dim * hidden1, 1, 1), MTLSizeMake(64, 1, 1));

  // --- sgd updates ---
  // update linear3 weights
  launch_kernel("sgd_update", {mnist_model.linear3, gradW3, lr_buf},
                MTLSizeMake(hidden2 * output_dim, 1, 1), MTLSizeMake(64, 1, 1));
  // update linear2 weights
  launch_kernel("sgd_update", {mnist_model.linear2, gradW2, lr_buf},
                MTLSizeMake(hidden1 * hidden2, 1, 1), MTLSizeMake(64, 1, 1));
  // update linear1 weights
  launch_kernel("sgd_update", {mnist_model.linear1, gradW1, lr_buf},
                MTLSizeMake(input_dim * hidden1, 1, 1), MTLSizeMake(64, 1, 1));

  // note: bias updates not implemented yet—they follow similar pattern

  return loss;
}

int main(void) {
  metal.init();
  mnist.init(metal.device);
  mnist_model.init();

  try {
    uint32_t num_images, rows, cols, num_labels;

    load_images("./mnist/train-images.idx3-ubyte", mnist.xtrain_un, num_images, rows, cols);
    load_labels("./mnist/train-labels.idx1-ubyte", mnist.ytrain, num_labels);
    load_images("./mnist/t10k-images.idx3-ubyte", mnist.xtest_un, num_images, rows, cols);
    load_labels("./mnist/t10k-labels.idx1-ubyte", mnist.ytest, num_labels);

    mnist.normalize();

    NSLog(@"loaded %u train images and %u test images", num_images, num_labels);
  } catch (const std::exception &e) {
    NSLog(@"error: %s", e.what());
    return 1;
  }

  const int steps_per_epoch = 60000 / 200; // assuming batch_size = 200
  const int num_epochs = 30;

  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float epoch_loss = 0.0f;
    Timer t;

    for (int step = 0; step < steps_per_epoch; ++step) {
      float loss = train(step);
      epoch_loss += loss;

      if (step % 100 == 0) {
        NSLog(@"[epoch %d] step %d/%d  loss = %.4f", epoch+1, step, steps_per_epoch, loss);
      }
    }

    NSLog(@"[epoch %d done] avg loss = %.4f (%.2f ms)", epoch+1, epoch_loss / steps_per_epoch, t.elapsed_ms());
  }

  return 0;
}
