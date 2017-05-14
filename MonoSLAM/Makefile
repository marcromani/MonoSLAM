# Compiler
CC = g++

# Source files
FILES_C = camera.cpp feature.cpp main.cpp map.cpp quaternion.cpp util.cpp

# Executable
TARGET = main

# Compilation options
#CFLAGS = -Wall -g -std=c++14 -march=native -O3 -funroll-loops -pthread -fprofile-generate `pkg-config --cflags opencv`
CFLAGS = -Wall -g -std=c++14 -march=native -O3 -funroll-loops -pthread -fprofile-use -fprofile-correction `pkg-config --cflags opencv`

# Linker options
#LFLAGS = -pthread -fprofile-generate `pkg-config --libs opencv`
LFLAGS = -pthread -fprofile-use -fprofile-correction `pkg-config --libs opencv`

FILES_O = $(subst .cpp,.o,$(FILES_C))

$(TARGET): $(FILES_O) Makefile
	$(CC) $(FILES_O) -o $(TARGET) $(LFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $<

all: $(TARGET)

clean:
	rm $(FILES_O) $(TARGET)