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
#include <stdio.h>
#include <unordered_map> 
#include <vector>        
#include <string>

using namespace std; 

struct Timing {
  std::string *label;
  chrono::steady_clock::time_point start;
  Timing(const char *label) : label(label), start(std::chrono::steady_clock::now()) {};
  ~Timing() {
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      printf("%s: %f ms\n", label, ms);
  }
}

struct metalcontext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  id<MTLCommandBuffer> cmd = [metal.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

  std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;

  void init() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw runtime_error("failed to create metal device.");
    NSLog(@"using device %@", [device name]);
    queue = [device newCommandQueue];
    if (!queue) throw runtime_error("failed to create metal command queue.");

    NSString *path = [[NSBundle mainBundle] pathForResource:@"kernels" ofType:@"metal"];
    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!src) throw std::runtime_error("failed to read metal source");

    NSError *err = nil;
    library = [device newLibraryWithSource:src options:nil error:&err];
    if (!library) {
        NSLog(@"metal compile error: %@", err);
        throw std::runtime_error("metal compile failed");
    }
    NSLog(@"metal ok");
  }

  id<MTLComputePipelineState> get_pipeline(const char *name) {
    auto it = pipeline_cache.find(name);
    if (it != pipeline_cache.end()) return it->second;

    NSError *err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@(name)];
    if (!fn) throw std::runtime_error("kernel not found");

    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!p) throw std::runtime_error("pipeline state creation failed");

    pipeline_cache[name] = p;
    return p; 
  }
  // run command queue 
  
}; static metalcontext metal;

struct kernel_launch {
  const char *name;
  MTLSize grid, block;
  std::vector<id<MTLBuffer>> buffers;
  bool encoded = false;

  kernel_launch(const char *name): name(name) {}

  ~kernel_launch() {
    if (!encoded) _encode(); // auto encode on scope exit
  }

  kernel_launch& with_buffer(id<MTLBuffer> buf) { buffers.push_back(buf); return *this; }
  kernel_launch& with_buffers(std::initializer_list<id<MTLBuffer>> bufs) {
    buffers.insert(buffers.end(), bufs.begin(), bufs.end());
    return *this;
  }
  kernel_launch& with_grid(MTLSize g) { grid = g; return *this; }
  kernel_launch& with_block(MTLSize b) { block = b; return *this; }

  void _encode() {
    [enc setComputePipelineState:metal.get_pipeline(name)];
    for (size_t i = 0; i < buffers.size(); ++i)
        [metal.enc setBuffer:buffers[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
  }
};

struct linear {
  id<MTLBuffer> weights;
  id<MTLBuffer> bias;
  // also store required gradients for backward pass

  linear() {

  }

  void forward() {

  }

  void backward() {

  }

  void update() {

  }
}
