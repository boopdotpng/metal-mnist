CXX = clang++
CXXFLAGS = -std=c++17 -ObjC++ -fobjc-arc
INCLUDES = -I metal-cpp
FRAMEWORKS = -framework Metal -framework Foundation
NVCC = nvcc
NVCCFLAGS = -std=c++17
CUDA_SRC = cuda/main.cu cuda/kernels.cu

ifeq ($(RELEASE),1)
    OPTFLAGS = -O2
else
    OPTFLAGS = -g
endif

TARGET = main

.PHONY: all fast slow clean cuda

all: fast

fast: fast/main.mm
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) $(FRAMEWORKS) $(OPTFLAGS) -o $(TARGET)

slow: slow/main.mm
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) $(FRAMEWORKS) $(OPTFLAGS) -o $(TARGET)

cuda:
	cd cuda && nvcc -std=c++17 -Wno-deprecated-gpu-targets main.cu -o ../main

clean:
	rm -f $(TARGET)
