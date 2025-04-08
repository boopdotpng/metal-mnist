# Metal MNIST: slow, fast, and from scratch

## quickstart
You need to download the MNIST dataset to `./mnist/`.

> [link to kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

You also need to download and extract Apple's metal-cpp to `./metal-cpp`. 

> [link to metal-cpp](https://developer.apple.com/metal/cpp/)


*The version used here is metal-cpp_macOS15.2_iOS18.2.zip.*

`make && ./main`

## fast 
`make` will compile the sped up and optimized program. `main.mm` uses `fastkernels.metal` which are essentially sped up and optimized versions of the kernels in `fastkernels.metal`. 


## slow 
`make slow` will compile the original (quite slow) program. `slow.mm` uses `kernels.metal`. I wrote these kernels while I was learning and just trying to get a prototype training. 

Here are the training results for the slow version (batch size 500): 
```
$ ./slow
2025-04-05 17:06:47.207 slow[40474:6728489] using device Apple M3 Pro
2025-04-05 17:06:47.209 slow[40474:6728489] Metal library compiled successfully.
2025-04-05 17:06:47.217 slow[40474:6728489] Model initialized.
2025-04-05 17:06:47.232 slow[40474:6728489] Loaded 60000 images (28x28) from ./mnist/train-images.idx3-ubyte
2025-04-05 17:06:47.233 slow[40474:6728489] Loaded 60000 labels from ./mnist/train-labels.idx1-ubyte
2025-04-05 17:06:47.235 slow[40474:6728489] Loaded 10000 images (28x28) from ./mnist/t10k-images.idx3-ubyte
2025-04-05 17:06:47.235 slow[40474:6728489] Loaded 10000 labels from ./mnist/t10k-labels.idx1-ubyte
2025-04-05 17:06:47.235 slow[40474:6728489] Normalizing training images...
2025-04-05 17:06:47.245 slow[40474:6728489] Normalizing test images...
2025-04-05 17:06:47.247 slow[40474:6728489] Normalization complete, raw buffers released.
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
training finished.
validation: accuracy = 91.22% (154.35 ms)
```


Along with the code being borderline unreadable (the train function is ugly), it is slow. There are a lot of performance enhancements left on the table: 
1. We don't cache MTLFunction 
2. The matmul kernel is naive and doesn't use threadgroup memory or tiling. (this is a big perf hit) 
3. We run one kernel at a time and block the cpu, which is inefficient. They should be lazily queued and then run in one shot at the end. 
4. Some kernels can be fused. (matmul + bias) and (softmax + cross entropy loss), for example.  
5. General readability. 
6. We use Shared GPU buffers everywhere. (tbd if making the buffers private makes it faster).

The fast version addresses all of these. 

!TODO: show fast version performance 

## roadmap 

After the fast version is sufficiently fast (comparing to mlx), 
