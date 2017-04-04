# Compiler
CC = g++

# Source files
FILES_C = calibration2.cpp

# Executable
TARGET = main

# Compilation options
CFLAGS = -Wall -g -std=c++14 -march=native -O3 -funroll-loops `pkg-config --cflags opencv`

# Linker options
LFLAGS = `pkg-config --libs opencv`

FILES_O = $(subst .cpp,.o,$(FILES_C))

$(TARGET): $(FILES_O) Makefile
	$(CC) $(FILES_O) -o $(TARGET) $(LFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $<

all: $(TARGET)

clean:
	rm $(FILES_O) $(TARGET)