# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -ObjC++
INCLUDES = -I metal-cpp

# Frameworks
FRAMEWORKS = -framework Metal -framework Foundation -fobjc-arc 

# Source and target
SRC = main.mm
TARGET = main

# Build target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) $(FRAMEWORKS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)

slow: slow.mm
	$(CXX) $(CXXFLAGS) slow.mm $(INCLUDES) $(FRAMEWORKS) -o slow
