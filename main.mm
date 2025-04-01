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
#include <unordered_map> // Include missing header
#include <string>        // Include missing header
#include <vector>        // Include missing header (though initializer_list is used)


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
    if (!device) {
        throw runtime_error("Failed to create Metal device.");
    }
    NSLog(@"using device %@", [device name]);
    queue = [device newCommandQueue];
    if (!queue) {
        throw runtime_error("Failed to create Metal command queue.");
    }

    NSError *err = nil;
    // Assuming kernels.metal is in the Resources folder of the application bundle
    NSString *path = [[NSBundle mainBundle] pathForResource:@"kernels" ofType:@"metal"];
    if (!path) {
        // Fallback path (e.g., running from command line next to source)
         path = @"kernels.metal";
         NSLog(@"Warning: kernels.metal not found in bundle, trying local path: %@", path);
    }

    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!src) {
        NSLog(@"failed to read metal source at %@: %@", path, err);
        throw runtime_error("metal source read failed");
    }

    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0; // Specify a language version

    library = [device newLibraryWithSource:src options:options error:&err];
    if (!library) {
        NSLog(@"metal compile error: %@", err);
        throw runtime_error("metal library compile failed");
    }
    NSLog(@"Metal library compiled successfully.");
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

  // Store counts for clarity
  uint32_t train_count = 0;
  uint32_t test_count = 0;
  uint32_t rows = 0;
  uint32_t cols = 0;
  const uint32_t pixels_per_image = 28*28;


  void init(id<MTLDevice> device) {
    // Allocate based on known MNIST sizes
    train_count = 60000;
    test_count = 10000;
    rows = 28;
    cols = 28;

    xtrain_un = [device newBufferWithLength:train_count * pixels_per_image options:MTLResourceStorageModeShared];
    xtest_un  = [device newBufferWithLength:test_count * pixels_per_image options:MTLResourceStorageModeShared];

    xtrain = [device newBufferWithLength:train_count * pixels_per_image * sizeof(float) options:MTLResourceStorageModeShared];
    xtest  = [device newBufferWithLength:test_count * pixels_per_image * sizeof(float) options:MTLResourceStorageModeShared];

    ytrain = [device newBufferWithLength:train_count * sizeof(unsigned char) options:MTLResourceStorageModeShared]; // Use uchar matching kernel
    ytest  = [device newBufferWithLength:test_count * sizeof(unsigned char) options:MTLResourceStorageModeShared];  // Use uchar matching kernel

    if (!xtrain_un || !xtest_un || !xtrain || !xtest || !ytrain || !ytest) {
        throw std::runtime_error("Failed to allocate MNIST data buffers.");
    }
  }

  void normalize() {
    if (!xtrain_un || !xtest_un) {
        NSLog(@"Raw data buffers already released or not loaded.");
        return;
    }
     NSLog(@"Normalizing training images...");
     // Correct grid size for normalization based on buffer length
    launch_kernel("normalize_image", {xtrain_un, xtrain}, MTLSizeMake(train_count * pixels_per_image, 1, 1), MTLSizeMake(256, 1, 1)); // Use larger block size?
    NSLog(@"Normalizing test images...");
    launch_kernel("normalize_image", {xtest_un, xtest}, MTLSizeMake(test_count * pixels_per_image, 1, 1), MTLSizeMake(256, 1, 1));

    // Release uint8 buffers after normalization as they are no longer needed
    xtrain_un = nil;
    xtest_un = nil;
    NSLog(@"Normalization complete, raw buffers released.");
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
    if (!fn) {
        NSString* errStr = [NSString stringWithFormat:@"Kernel function '%s' not found in library.", name];
        NSLog(@"%@", errStr);
        throw runtime_error([errStr UTF8String]);
    }
    pipeline = [metal.device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipeline) {
        NSLog(@"Failed to create pipeline state for function %s: %@", name, err);
        throw runtime_error("pipeline state creation failed");
    }
     // Check max threadgroup size
     NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
     if (block.width * block.height * block.depth > maxThreads) {
         NSLog(@"Warning: Requested threadgroup size (%lux%lux%lu = %lu) exceeds device max (%lu) for kernel %s",
               block.width, block.height, block.depth, block.width * block.height * block.depth, maxThreads, name);
         // Adjust block size or handle error - for now just log warning
         // block.width = maxThreads; // Example adjustment (needs careful thought)
     }
    pipeline_cache[name] = pipeline;
  }

  id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
   if (!cmd) { throw std::runtime_error("Failed to create command buffer."); }
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
   if (!enc) { throw std::runtime_error("Failed to create command encoder."); }

  [enc setComputePipelineState:pipeline];

  int i = 0;
  for(id<MTLBuffer> buf: buffers) {
    if (!buf) {
        NSString* errStr = [NSString stringWithFormat:@"Error: Buffer at index %d for kernel '%s' is nil.", i, name];
        NSLog(@"%@", errStr);
        throw runtime_error([errStr UTF8String]);
    }
    [enc setBuffer: buf offset:0 atIndex:i++];
  }

  // Ensure grid and block sizes are valid
    if (grid.width == 0 || grid.height == 0 || grid.depth == 0 ||
        block.width == 0 || block.height == 0 || block.depth == 0) {
        NSString* errStr = [NSString stringWithFormat:@"Error: Invalid grid or block size for kernel '%s'. Grid: (%lu, %lu, %lu), Block: (%lu, %lu, %lu)",
                name, grid.width, grid.height, grid.depth, block.width, block.height, block.depth];
        NSLog(@"%@", errStr);
        throw runtime_error([errStr UTF8String]);
    }


  [enc dispatchThreads:grid threadsPerThreadgroup:block];

  [enc endEncoding];
  [cmd commit];
  //[cmd waitUntilScheduled]; // Optional: wait until scheduled
  [cmd waitUntilCompleted]; // Wait for completion

  // Check for command buffer errors (optional but good practice)
    if (cmd.status == MTLCommandBufferStatusError) {
        NSLog(@"Error: Command buffer failed execution for kernel %s: %@", name, cmd.error);
        // Depending on the error, you might want to throw or handle it
    }
}

// class model
struct model {
  id<MTLBuffer> linear1;
  id<MTLBuffer> bias1;
  id<MTLBuffer> linear2;
  id<MTLBuffer> bias2;
  id<MTLBuffer> linear3;
  id<MTLBuffer> bias3;

   const uint32_t input_dim = 784;
   const uint32_t hidden1 = 512;
   const uint32_t hidden2 = 128;
   const uint32_t output_dim = 10;

  void init() {
    id<MTLBuffer> param_buf = [metal.device newBufferWithLength:sizeof(InitParams) options:MTLResourceStorageModeShared];
    InitParams p;
    MTLSize block = MTLSizeMake(128,1,1); // Reusable block size for init

    // Layer 1 Weights
    p = {input_dim * hidden1, input_dim, hidden1};
    memcpy(param_buf.contents, &p, sizeof(p));
    linear1 = [metal.device newBufferWithLength:input_dim * hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear1, param_buf}, MTLSizeMake(p.param_count,1,1), block);

    // Layer 2 Weights
    p = {hidden1 * hidden2, hidden1, hidden2};
    memcpy(param_buf.contents, &p, sizeof(p));
    linear2 = [metal.device newBufferWithLength:hidden1 * hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear2, param_buf}, MTLSizeMake(p.param_count,1,1), block);

    // Layer 3 Weights
    p = {hidden2 * output_dim, hidden2, output_dim};
    memcpy(param_buf.contents, &p, sizeof(p));
    linear3 = [metal.device newBufferWithLength:hidden2 * output_dim * sizeof(float) options:MTLResourceStorageModeShared];
    launch_kernel("init_random_weights", {linear3, param_buf}, MTLSizeMake(p.param_count,1,1), block);

    // --- Bias Initialization ---
    // Option 1: Use the same random init (as original code)
    // Option 2: Initialize to zero (often preferred) - Uncomment to use zero init

    // Bias 1
    p = {hidden1, hidden1, hidden1}; // Params for random init if used
    memcpy(param_buf.contents, &p, sizeof(p));
    bias1 = [metal.device newBufferWithLength:hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
    // launch_kernel("init_random_weights", {bias1, param_buf}, MTLSizeMake(p.param_count, 1, 1), block); // Random init
    memset(bias1.contents, 0, hidden1 * sizeof(float)); // Zero init

    // Bias 2
    p = {hidden2, hidden2, hidden2}; // Params for random init if used
    memcpy(param_buf.contents, &p, sizeof(p));
    bias2 = [metal.device newBufferWithLength:hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
    // launch_kernel("init_random_weights", {bias2, param_buf}, MTLSizeMake(p.param_count, 1, 1), block); // Random init
     memset(bias2.contents, 0, hidden2 * sizeof(float)); // Zero init


    // Bias 3
    p = {output_dim, output_dim, output_dim}; // Params for random init if used
    memcpy(param_buf.contents, &p, sizeof(p));
    bias3 = [metal.device newBufferWithLength:output_dim * sizeof(float) options:MTLResourceStorageModeShared];
    // launch_kernel("init_random_weights", {bias3, param_buf}, MTLSizeMake(p.param_count, 1, 1), MTLSizeMake(p.param_count > 0 ? p.param_count : 1, 1, 1)); // Random init, adjust block if needed
     memset(bias3.contents, 0, output_dim * sizeof(float)); // Zero init

    NSLog(@"Model initialized.");
  }
};
static model mnist_model;

// --- Data Loading (Mostly Unchanged, ensure paths are correct) ---
uint32_t read_big_endian_uint32(ifstream &ifs) {
  unsigned char bytes[4];
  if (!ifs.read((char*)bytes, 4)) {
       throw std::runtime_error("Failed to read 4 bytes from file stream.");
  }
  return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
         (uint32_t(bytes[2]) << 8)  | uint32_t(bytes[3]);
}

void load_images(const string &filename, id<MTLBuffer> buffer, uint32_t expected_count, uint32_t &num_images, uint32_t &rows, uint32_t &cols) {
  ifstream file(filename, ios::binary);
  if (!file) throw runtime_error("error opening image file: " + filename);

  uint32_t magic = read_big_endian_uint32(file);
   if (magic != 0x00000803) throw runtime_error("Invalid magic number in image file: " + filename);

  num_images = read_big_endian_uint32(file);
  rows = read_big_endian_uint32(file);
  cols = read_big_endian_uint32(file);

  if (num_images != expected_count || rows != 28 || cols != 28) {
       throw runtime_error("Mismatch in image file header: " + filename);
   }

  size_t data_size = num_images * rows * cols;
  if (buffer.length < data_size) {
        throw runtime_error("Buffer too small for image data in: " + filename);
  }

  void *dest = buffer.contents;
  if (!file.read((char*)dest, data_size)) {
      throw runtime_error("Failed to read image data from: " + filename);
  }
  NSLog(@"Loaded %u images (%ux%u) from %s", num_images, rows, cols, filename.c_str());
}

void load_labels(const string &filename, id<MTLBuffer> buffer, uint32_t expected_count, uint32_t &num_labels) {
  ifstream file(filename, ios::binary);
  if (!file) throw runtime_error("error opening label file: " + filename);

  uint32_t magic = read_big_endian_uint32(file);
  if (magic != 0x00000801) throw runtime_error("Invalid magic number in label file: " + filename);

  num_labels = read_big_endian_uint32(file);
   if (num_labels != expected_count) {
       throw runtime_error("Mismatch in label file header: " + filename);
   }

   size_t data_size = num_labels * sizeof(unsigned char); // Labels are bytes
   if (buffer.length < data_size) {
        throw runtime_error("Buffer too small for label data in: " + filename);
   }

  void *dest = buffer.contents;
  if (!file.read((char*)dest, data_size)) {
      throw runtime_error("Failed to read label data from: " + filename);
  }
  NSLog(@"Loaded %u labels from %s", num_labels, filename.c_str());
}
// --- End Data Loading ---


float train(uint32_t batch_index) {
  const uint32_t batch_size = 200;
  // Use dimensions from model struct
  const uint32_t input_dim = mnist_model.input_dim;
  const uint32_t hidden1 = mnist_model.hidden1;
  const uint32_t hidden2 = mnist_model.hidden2;
  const uint32_t output_dim = mnist_model.output_dim;
  float lr = 0.01f;

  // Reusable buffers (consider making static or member if train is in a class)
  static id<MTLBuffer> shape_buf = [metal.device newBufferWithLength:sizeof(MatmulParams) options:MTLResourceStorageModeShared];
  static id<MTLBuffer> offset_buf = [metal.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  static id<MTLBuffer> bias_dim_buf = [metal.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared]; // Use uint32_t for consistency?
  static id<MTLBuffer> lr_buf = [metal.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
  static bool static_bufs_initialized = false;
  if (!static_bufs_initialized) {
      memcpy(lr_buf.contents, &lr, sizeof(float));
      if (!shape_buf || !offset_buf || !bias_dim_buf || !lr_buf) throw std::runtime_error("Failed to alloc static train buffers");
      static_bufs_initialized = true;
  }

  // Per-batch buffers
  id<MTLBuffer> batch = [metal.device newBufferWithLength:batch_size * input_dim * sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> batch_labels = [metal.device newBufferWithLength:batch_size * sizeof(unsigned char) options:MTLResourceStorageModeShared];
  id<MTLBuffer> losses_buf = [metal.device newBufferWithLength:batch_size * sizeof(float) options:MTLResourceStorageModeShared];
   if (!batch || !batch_labels || !losses_buf) throw std::runtime_error("Failed to alloc per-batch train buffers");


  // --- Copy Batch Data ---
  uint32_t offset = batch_index * batch_size;
  memcpy([offset_buf contents], &offset, sizeof(uint32_t));

  MTLSize copy_block = MTLSizeMake(256, 1, 1); // Block size for copy kernels
  launch_kernel("copy_batch", {mnist.xtrain, batch, offset_buf},
                MTLSizeMake(batch_size * input_dim, 1, 1), copy_block);
  launch_kernel("copy_batch_labels", {mnist.ytrain, batch_labels, offset_buf},
                MTLSizeMake(batch_size, 1, 1), copy_block);

  // ------------------
  // FORWARD PASS
  // ------------------
   MTLSize matmul_block = MTLSizeMake(128, 1, 1); // Block for matmul, etc. Check device limits
   MTLSize bias_relu_block = MTLSizeMake(128, 1, 1);


  // Layer 1: [B x 784] * [784 x 512] + bias -> ReLU -> [B x 512]
  // Allocate separate buffers for pre-ReLU and post-ReLU
  id<MTLBuffer> buf1_pre_relu = [metal.device newBufferWithLength:batch_size * hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf1_relu = [metal.device newBufferWithLength:batch_size * hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape1 = {batch_size, input_dim, hidden1};
  memcpy([shape_buf contents], &shape1, sizeof(shape1));
  launch_kernel("matmul", {batch, mnist_model.linear1, buf1_pre_relu, shape_buf},
                MTLSizeMake(batch_size * hidden1, 1, 1), matmul_block);

  uint32_t d1 = hidden1; // Use uint32_t to match kernel expected type if changed
  memcpy(bias_dim_buf.contents, &d1, sizeof(d1));
  launch_kernel("add_bias", {buf1_pre_relu, mnist_model.bias1, bias_dim_buf},
                MTLSizeMake(batch_size * hidden1, 1, 1), bias_relu_block);
  // Apply ReLU out-of-place
  launch_kernel("relu", {buf1_pre_relu, buf1_relu},
                MTLSizeMake(batch_size * hidden1, 1, 1), bias_relu_block);


  // Layer 2: [B x 512] * [512 x 128] + bias -> ReLU -> [B x 128]
  id<MTLBuffer> buf2_pre_relu = [metal.device newBufferWithLength:batch_size * hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf2_relu = [metal.device newBufferWithLength:batch_size * hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape2 = {batch_size, hidden1, hidden2};
  memcpy([shape_buf contents], &shape2, sizeof(shape2));
  launch_kernel("matmul", {buf1_relu, mnist_model.linear2, buf2_pre_relu, shape_buf}, // Input is buf1_relu
                MTLSizeMake(batch_size * hidden2, 1, 1), matmul_block);

  uint32_t d2 = hidden2;
  memcpy(bias_dim_buf.contents, &d2, sizeof(d2));
  launch_kernel("add_bias", {buf2_pre_relu, mnist_model.bias2, bias_dim_buf},
                MTLSizeMake(batch_size * hidden2, 1, 1), bias_relu_block);
  // Apply ReLU out-of-place
  launch_kernel("relu", {buf2_pre_relu, buf2_relu},
                MTLSizeMake(batch_size * hidden2, 1, 1), bias_relu_block);


  // Layer 3 (Output): [B x 128] * [128 x 10] + bias -> Softmax -> [B x 10]
  id<MTLBuffer> buf3_logits = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float) options:MTLResourceStorageModeShared];
  MatmulParams shape3 = {batch_size, hidden2, output_dim};
  memcpy([shape_buf contents], &shape3, sizeof(shape3));
  launch_kernel("matmul", {buf2_relu, mnist_model.linear3, buf3_logits, shape_buf}, // Input is buf2_relu
                MTLSizeMake(batch_size * output_dim, 1, 1), matmul_block);

  uint32_t d3 = output_dim;
  memcpy(bias_dim_buf.contents, &d3, sizeof(d3));
  launch_kernel("add_bias", {buf3_logits, mnist_model.bias3, bias_dim_buf},
                MTLSizeMake(batch_size * output_dim, 1, 1), bias_relu_block);

  id<MTLBuffer> probs = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float) options:MTLResourceStorageModeShared];
  launch_kernel("softmax", {buf3_logits, probs, bias_dim_buf}, // Pass dim buffer
              MTLSizeMake(batch_size * output_dim, 1, 1), bias_relu_block);


  // --- Loss Computation ---
  launch_kernel("cross_entropy_loss", {probs, batch_labels, losses_buf, bias_dim_buf}, // Pass dim buffer
                MTLSizeMake(batch_size, 1, 1), bias_relu_block); // Grid is per-sample

  float loss_sum = 0;
  // Ensure buffer data is synchronized if needed (waitUntilCompleted does this)
  float *losses_ptr = (float *)losses_buf.contents;
  for (int i = 0; i < batch_size; ++i) {
    loss_sum += losses_ptr[i];
  }
  float loss = loss_sum / batch_size;

  // ------------------
  // BACKWARD PASS
  // ------------------
   MTLSize backward_block = MTLSizeMake(128, 1, 1); // Check device limits

  // Gradient of loss w.r.t. logits (output of layer 3 before softmax)
  // dL_dlogits = probs - y_one_hot
  id<MTLBuffer> dL_dlogits = [metal.device newBufferWithLength:batch_size * output_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
  launch_kernel("softmax_cross_entropy_backward", {probs, batch_labels, dL_dlogits, bias_dim_buf}, // Pass dim buf
                MTLSizeMake(batch_size * output_dim, 1, 1), backward_block);

  // --- Layer 3 Backward ---
  // gradW3 = (buf2_relu)ᵀ @ dL_dlogits
  // gradB3 = sum(dL_dlogits, axis=0) / batch_size
  // dL_dbuf2_relu = dL_dlogits @ (linear3)ᵀ

  // Weight Gradient (W3)
  memcpy([shape_buf contents], &shape3, sizeof(shape3)); // Use forward shape
  id<MTLBuffer> gradW3 = [metal.device newBufferWithLength:hidden2 * output_dim * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {buf2_relu, dL_dlogits, gradW3, shape_buf}, // Input act: buf2_relu
                MTLSizeMake(hidden2 * output_dim, 1, 1), backward_block);

  // Bias Gradient (B3)
  id<MTLBuffer> gradB3 = [metal.device newBufferWithLength:output_dim * sizeof(float) options:MTLResourceStorageModeShared];
  // Need batch size buffer - create once if possible
   id<MTLBuffer> batch_size_buf = [metal.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
   uint32_t batch_size_u32 = batch_size;
   memcpy(batch_size_buf.contents, &batch_size_u32, sizeof(uint32_t));
   memcpy(bias_dim_buf.contents, &d3, sizeof(d3)); // Ensure correct dim is set
  launch_kernel("bias_grad_sum", {dL_dlogits, gradB3, batch_size_buf, bias_dim_buf},
                MTLSizeMake(output_dim, 1, 1), backward_block); // Grid is per bias element

  // Input Gradient (dL propagated to buf2_relu)
  id<MTLBuffer> dL_dbuf2_relu = [metal.device newBufferWithLength:batch_size * hidden2 * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_input", {dL_dlogits, mnist_model.linear3, dL_dbuf2_relu, shape_buf},
                MTLSizeMake(batch_size * hidden2, 1, 1), backward_block);

  // SGD Update (Layer 3)
  launch_kernel("sgd_update", {mnist_model.linear3, gradW3, lr_buf},
                MTLSizeMake(hidden2 * output_dim, 1, 1), backward_block);
  launch_kernel("sgd_update", {mnist_model.bias3, gradB3, lr_buf},
                MTLSizeMake(output_dim, 1, 1), backward_block);


  // --- Layer 2 Backward ---
  // dL_dbuf2_pre_relu = dL_dbuf2_relu * (buf2_pre_relu > 0)
  // gradW2 = (buf1_relu)ᵀ @ dL_dbuf2_pre_relu
  // gradB2 = sum(dL_dbuf2_pre_relu, axis=0) / batch_size
  // dL_dbuf1_relu = dL_dbuf2_pre_relu @ (linear2)ᵀ

  // ReLU Backward (use pre-relu buffer for condition, modify gradient in-place)
  // Creates dL_dbuf2_pre_relu by modifying dL_dbuf2_relu
  launch_kernel("relu_backward", {buf2_pre_relu, dL_dbuf2_relu}, // Input: buf2_pre_relu, GradInOut: dL_dbuf2_relu
                MTLSizeMake(batch_size * hidden2, 1, 1), backward_block);
  // Now dL_dbuf2_relu holds the gradient w.r.t the pre-relu activation

  // Weight Gradient (W2)
  memcpy([shape_buf contents], &shape2, sizeof(shape2)); // Use forward shape
  id<MTLBuffer> gradW2 = [metal.device newBufferWithLength:hidden1 * hidden2 * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {buf1_relu, dL_dbuf2_relu, gradW2, shape_buf}, // Input act: buf1_relu, Grad: dL_dbuf2_pre_relu (now in dL_dbuf2_relu)
                MTLSizeMake(hidden1 * hidden2, 1, 1), backward_block);

   // Bias Gradient (B2)
  id<MTLBuffer> gradB2 = [metal.device newBufferWithLength:hidden2 * sizeof(float) options:MTLResourceStorageModeShared];
  memcpy(bias_dim_buf.contents, &d2, sizeof(d2)); // Set correct dim
  // batch_size_buf already contains the correct batch size
  launch_kernel("bias_grad_sum", {dL_dbuf2_relu, gradB2, batch_size_buf, bias_dim_buf}, // Grad: dL_dbuf2_pre_relu
                MTLSizeMake(hidden2, 1, 1), backward_block);

  // Input Gradient (dL propagated to buf1_relu)
  id<MTLBuffer> dL_dbuf1_relu = [metal.device newBufferWithLength:batch_size * hidden1 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_input", {dL_dbuf2_relu, mnist_model.linear2, dL_dbuf1_relu, shape_buf}, // Grad: dL_dbuf2_pre_relu
                MTLSizeMake(batch_size * hidden1, 1, 1), backward_block);

  // SGD Update (Layer 2)
  launch_kernel("sgd_update", {mnist_model.linear2, gradW2, lr_buf},
                MTLSizeMake(hidden1 * hidden2, 1, 1), backward_block);
  launch_kernel("sgd_update", {mnist_model.bias2, gradB2, lr_buf},
                MTLSizeMake(hidden2, 1, 1), backward_block);


  // --- Layer 1 Backward ---
  // dL_dbuf1_pre_relu = dL_dbuf1_relu * (buf1_pre_relu > 0)
  // gradW1 = (batch)ᵀ @ dL_dbuf1_pre_relu
  // gradB1 = sum(dL_dbuf1_pre_relu, axis=0) / batch_size
  // dL_dbatch = dL_dbuf1_pre_relu @ (linear1)ᵀ (Not needed for update)

  // ReLU Backward
  launch_kernel("relu_backward", {buf1_pre_relu, dL_dbuf1_relu}, // Input: buf1_pre_relu, GradInOut: dL_dbuf1_relu
                MTLSizeMake(batch_size * hidden1, 1, 1), backward_block);
  // Now dL_dbuf1_relu holds the gradient w.r.t the pre-relu activation

  // Weight Gradient (W1)
  memcpy([shape_buf contents], &shape1, sizeof(shape1)); // Use forward shape
  id<MTLBuffer> gradW1 = [metal.device newBufferWithLength:input_dim * hidden1 * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  launch_kernel("matmul_grad_w", {batch, dL_dbuf1_relu, gradW1, shape_buf}, // Input act: batch, Grad: dL_dbuf1_pre_relu
                MTLSizeMake(input_dim * hidden1, 1, 1), backward_block);

  // Bias Gradient (B1)
  id<MTLBuffer> gradB1 = [metal.device newBufferWithLength:hidden1 * sizeof(float) options:MTLResourceStorageModeShared];
  memcpy(bias_dim_buf.contents, &d1, sizeof(d1)); // Set correct dim
  // batch_size_buf already contains the correct batch size
  launch_kernel("bias_grad_sum", {dL_dbuf1_relu, gradB1, batch_size_buf, bias_dim_buf}, // Grad: dL_dbuf1_pre_relu
                MTLSizeMake(hidden1, 1, 1), backward_block);

  // SGD Update (Layer 1)
  launch_kernel("sgd_update", {mnist_model.linear1, gradW1, lr_buf},
                MTLSizeMake(input_dim * hidden1, 1, 1), backward_block);
  launch_kernel("sgd_update", {mnist_model.bias1, gradB1, lr_buf},
                MTLSizeMake(hidden1, 1, 1), backward_block);

  // Note: Intermediate buffers (pre_relu, grads, etc.) will be released automatically
  // when they go out of scope (ARC for Objective-C objects).

  return loss;
}

int main(void) {
 @autoreleasepool { // Good practice for Obj-C objects
    try {
        metal.init();
        mnist.init(metal.device);
        mnist_model.init(); // Init model dimensions before loading

        // --- Load Data ---
        // Ensure the paths are correct relative to your executable
        // Or provide absolute paths.
        const string data_path = "./mnist/"; // Adjust if needed
        uint32_t num_images_train, rows_train, cols_train, num_labels_train;
        uint32_t num_images_test, rows_test, cols_test, num_labels_test;

        load_images(data_path + "train-images.idx3-ubyte", mnist.xtrain_un, mnist.train_count, num_images_train, rows_train, cols_train);
        load_labels(data_path + "train-labels.idx1-ubyte", mnist.ytrain, mnist.train_count, num_labels_train);
        load_images(data_path + "t10k-images.idx3-ubyte", mnist.xtest_un, mnist.test_count, num_images_test, rows_test, cols_test);
        load_labels(data_path + "t10k-labels.idx1-ubyte", mnist.ytest, mnist.test_count, num_labels_test);

        mnist.normalize(); // Normalize after loading

        // --- Training Loop ---
        const int batch_size = 200; // Must match the value in train()
        if (mnist.train_count == 0) {
             throw runtime_error("Training data count is zero, cannot train.");
        }
        const int steps_per_epoch = mnist.train_count / batch_size;
        const int num_epochs = 10; // Reduced for quicker testing initially

        NSLog(@"Starting training: %d epochs, %d steps/epoch, batch size %d", num_epochs, steps_per_epoch, batch_size);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;
            Timer t; // Start timer for epoch

            for (int step = 0; step < steps_per_epoch; ++step) {
                float batch_loss = train(step); // Pass step index
                epoch_loss += batch_loss;

                 if (step % 100 == 99 || step == steps_per_epoch - 1) { // Log every 100 steps and last step
                     float avg_loss_so_far = epoch_loss / (step + 1);
                     NSLog(@"[Epoch %d/%d] Step %d/%d | Batch Loss: %.4f | Avg Epoch Loss: %.4f",
                           epoch + 1, num_epochs, step + 1, steps_per_epoch, batch_loss, avg_loss_so_far);
                 }
            }

            double epoch_time_ms = t.elapsed_ms();
             float avg_epoch_loss = epoch_loss / steps_per_epoch;
            NSLog(@"[Epoch %d Done] Avg Loss = %.4f (%.2f ms, %.2f ms/step)",
                  epoch + 1, avg_epoch_loss, epoch_time_ms, epoch_time_ms / steps_per_epoch);

             // Optional: Add evaluation on test set here after each epoch
        }

         NSLog(@"Training finished.");

    } catch (const std::exception &e) {
        // Log standard exceptions using NSLog for consistency
        NSString *errMsg = [NSString stringWithUTF8String:e.what()];
        NSLog(@"Runtime Error: %@", errMsg);
        return 1;
    } catch (...) {
        NSLog(@"An unknown error occurred.");
         return 1;
    }
 } // end @autoreleasepool
  return 0;
}
