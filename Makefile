# Compiler
CC = /opt/intel/oneapi/compiler/latest/bin/icx
#CC = gcc

# Compiler flags
CFLAGS = -lm

# Target executable
TARGET = neural_network

# Source files
SRCS = main.c

# Compile source files into object files
all: $(SRCS)
	$(CC) -o $(TARGET) $(SRCS) $(CFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET) network_weights.bin

# Run the program
run: $(TARGET)
	./$(TARGET)

# Do everything
doit: $(SRCS)
	rm -f $(TARGET) network_weights.bin && $(CC) -o $(TARGET) $(SRCS) $(CFLAGS) && ./$(TARGET)