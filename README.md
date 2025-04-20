# Metal MNIST: slow, fast, and from scratch
This is a from-scratch MNIST trainer in Metal. An educational tool to show how you run things on the gpu, and the math behind the forward and backward pass of simple networks.

## Quickstart
You need to download the MNIST dataset to `./mnist/`.

> [link to kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

You also need to download and extract Apple's metal-cpp to `./metal-cpp`. 

> [link to metal-cpp](https://developer.apple.com/metal/cpp/) <br>
> *version used: metal-cpp_macOS15.2_iOS18.2.zip.*

`make && ./main` (this runs the fast version by default).

## Slow 
The slow version is a naive implementation: kernels aren’t fused, there’s no caching, and everything runs one-by-one with manual waits. It’s functionally correct but inefficient.

`make slow` will compile the files under `slow`. 

Sample run (batch size = 500):
```
$ ./slow
[epoch 1 done] avg loss = 2.0363 (979.03 ms)
[epoch 2 done] avg loss = 1.4358 (947.31 ms)
[epoch 3 done] avg loss = 0.9308 (940.17 ms)
[epoch 4 done] avg loss = 0.6762 (948.64 ms)
[epoch 5 done] avg loss = 0.5533 (933.55 ms)
[epoch 6 done] avg loss = 0.4834 (898.02 ms)
[epoch 7 done] avg loss = 0.4375 (897.45 ms)
[epoch 8 done] avg loss = 0.4042 (931.32 ms)
[epoch 9 done] avg loss = 0.3784 (929.14 ms)
[epoch 10 done] avg loss = 0.3576 (935.85 ms)
validation: accuracy = 91.22% (154.35 ms)
```

Besides the code being borderline unreadable (the train function is ugly), it is slow. There are a lot of performance enhancements left on the table: 
1. We don't cache MTLFunction 
2. The matmul kernel is naive and doesn't use threadgroup memory or tiling. (this is a big perf hit) 
3. We run one kernel at a time and wait for it to finish, which is inefficient. They should be lazily queued and then run in one shot at the end. 
4. Some kernels can be fused. (matmul + bias) and (softmax + cross entropy loss), for example.  
5. General readability. 
6. Unnecessary allocations everywhere.

The fast version addresses most of these concerns. 

## Fast 

```
$ ./main
epoch loss: 1.759361, took 640.840583ms
epoch loss: 0.968614, took 625.226500ms
epoch loss: 0.672783, took 626.819959ms
epoch loss: 0.547242, took 626.391833ms
epoch loss: 0.478559, took 625.749292ms
epoch loss: 0.434771, took 626.470750ms
epoch loss: 0.404206, took 626.171958ms
epoch loss: 0.381349, took 625.446500ms
epoch loss: 0.363377, took 626.201083ms
epoch loss: 0.348680, took 627.181000ms
```
It's about 34% faster than the slow version, but there's still more to be done. 
1. Fuse backward kernels together.
2. Faster, tiled matmul.
3. Implement dataset offsets instead of copying slices

## Roadmap 
I want to get within 25% of [MLX](https://github.com/ml-explore/mlx). 
```
Epoch 0: Test accuracy 0.812, Time 0.316 (s)
Epoch 1: Test accuracy 0.877, Time 0.281 (s)
Epoch 2: Test accuracy 0.904, Time 0.280 (s)
Epoch 3: Test accuracy 0.917, Time 0.278 (s)
Epoch 4: Test accuracy 0.926, Time 0.276 (s)
Epoch 5: Test accuracy 0.918, Time 0.277 (s)
Epoch 6: Test accuracy 0.936, Time 0.275 (s)
Epoch 7: Test accuracy 0.921, Time 0.276 (s)
Epoch 8: Test accuracy 0.942, Time 0.275 (s)
Epoch 9: Test accuracy 0.941, Time 0.276 (s)
```
We're currently ~2x slower than MLX, but closing the gap.
Since Metal's profiling tools are limited (and profiling compute is barely supported), I'm going to port this to CUDA, analyze the kernel performance, and then apply the findings to metal and see how much faster it can be. 

It's very unlikely that I can beat MLX/tinygrad without writing cursed kernels, but this is more for educational purposes than raw performance / throughput. 

Convolutions are on the roadmap after matmul becomes fast. (Convs are just matmuls)
