# Compiler
CC = g++

# Source files
FILES_C = camera.cpp feature.cpp main.cpp map.cpp quaternion.cpp util.cpp

# Executable
TARGET = main

# Compilation options
CFLAGS = -Wall -g -std=c++17 -march=native -O3 -funroll-loops -pthread `pkg-config --cflags opencv`
# -Wall -Wextra -Wpedantic \
# -Wconversion -Wsign-conversion -Wfloat-conversion \
# -Wshadow -Wreorder -Wcast-align -Wnull-dereference \
# -Wdouble-promotion -Wformat=2 -Wundef -Wpointer-arith

# Linker options
LFLAGS = -pthread `pkg-config --libs opencv`

FILES_O = $(subst .cpp,.o,$(FILES_C))

$(TARGET): $(FILES_O) Makefile
    $(CC) $(FILES_O) -o $(TARGET) $(LFLAGS)

%.o: %.cpp Makefile
    $(CC) $(CFLAGS) -c $<

all: $(TARGET)

clean:
    rm $(FILES_O) $(TARGET)
