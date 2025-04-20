# Compiler and base flags
CXX = clang++
CXXFLAGS = -std=c++17 -ObjC++ -fobjc-arc
INCLUDES = -I metal-cpp
FRAMEWORKS = -framework Metal -framework Foundation

# Determine build type
ifeq ($(RELEASE),1)
    OPTFLAGS = -O2
else
    OPTFLAGS = -g
endif

# Output binary
TARGET = main

.PHONY: all fast slow clean

# Default target
all: fast

# Fast build
fast: fast/main.mm
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) $(FRAMEWORKS) $(OPTFLAGS) -o $(TARGET)

# Slow build
slow: slow/main.mm
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) $(FRAMEWORKS) $(OPTFLAGS) -o $(TARGET)

# Clean
clean:
	rm -f $(TARGET)

