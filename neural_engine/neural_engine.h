#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Enable SIMD intrinsics based on platform
#if defined(__SSE__) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#define USE_SIMD 1
#endif

// ANSI escape codes for colors
#define RESET   "\x1b[0m"
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[37m"

// Cache line size for alignment (typical value for modern CPUs)
#define CACHE_LINE_SIZE 64

/**
 * @struct Layers
 * @brief Represents a single layer in the neural network
 *
 * @member length Number of neurons in this layer
 * @member activation Pointer to the activation function used in this layer
 * @member activation_derivative Pointer to the derivative of the activation function
 *                             used for backpropagation
 */
struct Layers
{
  int length;
  float (*activation)(float);
  float (*activation_derivative)(float);
};

/**
 * @struct Network
 * @brief Represents the entire neural network structure
 *
 * @member layers Array of layer configurations
 * @member size Total number of layers in the network
 * @member weights 3D array of network weights:
 *                - First dimension: layer index
 *                - Second dimension: input neuron index (including bias)
 *                - Third dimension: output neuron index
 * @member cache Pre-allocated memory for forward and backward passes
 */
struct Network
{
  struct Layers *layers;
  int size;
  float ***weights;
  
  // Cache for intermediate values to avoid repeated allocations
  float **activations;        // [layer][neuron]
  float **errors;             // [layer][neuron]
  float **pre_activations;    // [layer][neuron] - values before activation func
};

/**
 * @brief Creates and initializes a new neural network with memory caching
 * @param layers Array of layer configurations
 * @param length Number of layers in the network
 * @return Initialized Network structure
 */
struct Network createNetwork(struct Layers layers[], int length)
{
  struct Network net;

  net.layers = layers;
  net.size = length;

  // Allocate memory for layers weights with alignment for better cache performance
  net.weights = (float ***)malloc((length - 1) * sizeof(float **));

  for (int layer = 0; layer < length - 1; layer++)
  {
    net.weights[layer] = (float **)malloc((layers[layer].length + 1) * sizeof(float *));

    for (int input_neuron = 0; input_neuron < layers[layer].length + 1; input_neuron++)
    {
      // Align memory to cache line for better performance
      size_t size = layers[layer + 1].length * sizeof(float);
      #ifdef USE_SIMD
      // For SIMD, align to 16-byte boundary
      net.weights[layer][input_neuron] = (float *)_mm_malloc(size, 16);
      #else
      net.weights[layer][input_neuron] = (float *)malloc(size);
      #endif
    }
  }

  // Pre-allocate memory for activations and errors (cache)
  net.activations = (float **)malloc(length * sizeof(float *));
  net.errors = (float **)malloc(length * sizeof(float *));
  net.pre_activations = (float **)malloc(length * sizeof(float *));

  for (int layer = 0; layer < length; layer++)
  {
    #ifdef USE_SIMD
    // For SIMD, align to 16-byte boundary
    net.activations[layer] = (float *)_mm_malloc(layers[layer].length * sizeof(float), 16);
    net.errors[layer] = (float *)_mm_malloc(layers[layer].length * sizeof(float), 16);
    net.pre_activations[layer] = (float *)_mm_malloc(layers[layer].length * sizeof(float), 16);
    #else
    net.activations[layer] = (float *)malloc(layers[layer].length * sizeof(float));
    net.errors[layer] = (float *)malloc(layers[layer].length * sizeof(float));
    net.pre_activations[layer] = (float *)malloc(layers[layer].length * sizeof(float));
    #endif
    
    // Initialize to zero
    memset(net.activations[layer], 0, layers[layer].length * sizeof(float));
    memset(net.errors[layer], 0, layers[layer].length * sizeof(float));
    memset(net.pre_activations[layer], 0, layers[layer].length * sizeof(float));
  }

  return net;
}

/**
 * @brief Checks if a file exists at the given path
 * @param filename Path to the file to check
 * @return 1 if file exists, 0 otherwise
 */
static inline int file_exists(const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (file)
  {
    fclose(file);
    return 1;
  }
  return 0;
}

/**
 * @brief Saves the network weights to a binary file
 * @param net Pointer to the network structure
 * @param filename Path where weights should be saved
 * @return 1 if successful, 0 if failed
 */
int saveWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer].length + 1; i++)
    {
      fwrite(net->weights[layer][i], sizeof(float), net->layers[layer + 1].length, file);
    }
  }
  fclose(file);
  return 1;
}

/**
 * @brief Loads network weights from a binary file
 * @param net Pointer to the network structure
 * @param filename Path to the weights file
 * @return 1 if successful, 0 if failed
 */
int loadWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (!file)
    return 0;

  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer].length + 1; i++)
    {
      fread(net->weights[layer][i], sizeof(float), net->layers[layer + 1].length, file);
    }
  }
  fclose(file);
  return 1;
}

/**
 * @brief Initializes network weights with random values using Xavier/Glorot initialization
 * @param net Pointer to the network structure
 */
void randomizeWeights(struct Network *net)
{
  srand(time(NULL));
  for (int layer = 0; layer < net->size - 1; layer++)
  {
    // Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (net->layers[layer].length + net->layers[layer + 1].length));

    for (int i = 0; i < net->layers[layer].length + 1; i++)
    {
      for (int j = 0; j < net->layers[layer + 1].length; j++)
      {
        float r = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        net->weights[layer][i][j] = r * scale;
      }
    }
  }
}

/**
 * @brief Optimized forward propagation with cached memory and SIMD where available
 * @param net Network structure
 * @param input Array of input values
 * @return Pointer to output layer activations (no memory allocation)
 */
static inline float *forward_propagate(struct Network *net, float input[])
{
  // Copy input to first layer activations
  memcpy(net->activations[0], input, net->layers[0].length * sizeof(float));

  // Forward pass through each layer
  for (int layer_idx = 0; layer_idx < net->size - 1; layer_idx++)
  {
    const int input_size = net->layers[layer_idx].length;
    const int output_size = net->layers[layer_idx + 1].length;
    
    // Zero out pre-activations for the next layer
    memset(net->pre_activations[layer_idx + 1], 0, output_size * sizeof(float));
    
    // Matrix multiplication with loop unrolling for better performance
    // Process 4 output neurons at once when possible
    int output_blocks = output_size / 4;
    int output_remainder = output_size % 4;
    
    #ifdef USE_SIMD
    // SIMD implementation
    for (int i = 0; i < input_size; i++) {
      float activation = net->activations[layer_idx][i];
      __m128 activation_vec = _mm_set1_ps(activation);
      
      for (int j = 0; j < output_blocks; j++) {
        int out_idx = j * 4;
        __m128 weights_vec = _mm_loadu_ps(&net->weights[layer_idx][i][out_idx]);
        __m128 current_vec = _mm_loadu_ps(&net->pre_activations[layer_idx + 1][out_idx]);
        __m128 result = _mm_add_ps(current_vec, _mm_mul_ps(activation_vec, weights_vec));
        _mm_storeu_ps(&net->pre_activations[layer_idx + 1][out_idx], result);
      }
      
      // Handle remaining outputs
      for (int j = output_blocks * 4; j < output_size; j++) {
        net->pre_activations[layer_idx + 1][j] += activation * net->weights[layer_idx][i][j];
      }
    }
    
    // Add bias terms (bias is the last row in weights)
    for (int j = 0; j < output_blocks; j++) {
      int out_idx = j * 4;
      __m128 bias_vec = _mm_loadu_ps(&net->weights[layer_idx][input_size][out_idx]);
      __m128 current_vec = _mm_loadu_ps(&net->pre_activations[layer_idx + 1][out_idx]);
      _mm_storeu_ps(&net->pre_activations[layer_idx + 1][out_idx], _mm_add_ps(current_vec, bias_vec));
    }
    
    // Handle remaining bias terms
    for (int j = output_blocks * 4; j < output_size; j++) {
      net->pre_activations[layer_idx + 1][j] += net->weights[layer_idx][input_size][j];
    }
    #else
    // Non-SIMD implementation with loop unrolling
    for (int i = 0; i < input_size; i++) {
      float activation = net->activations[layer_idx][i];
      
      // Process 4 outputs at once
      for (int j = 0; j < output_blocks * 4; j += 4) {
        net->pre_activations[layer_idx + 1][j]   += activation * net->weights[layer_idx][i][j];
        net->pre_activations[layer_idx + 1][j+1] += activation * net->weights[layer_idx][i][j+1];
        net->pre_activations[layer_idx + 1][j+2] += activation * net->weights[layer_idx][i][j+2];
        net->pre_activations[layer_idx + 1][j+3] += activation * net->weights[layer_idx][i][j+3];
      }
      
      // Handle remaining outputs
      for (int j = output_blocks * 4; j < output_size; j++) {
        net->pre_activations[layer_idx + 1][j] += activation * net->weights[layer_idx][i][j];
      }
    }
    
    // Add bias terms (bias is the last row in weights)
    for (int j = 0; j < output_blocks * 4; j += 4) {
      net->pre_activations[layer_idx + 1][j]   += net->weights[layer_idx][input_size][j];
      net->pre_activations[layer_idx + 1][j+1] += net->weights[layer_idx][input_size][j+1];
      net->pre_activations[layer_idx + 1][j+2] += net->weights[layer_idx][input_size][j+2];
      net->pre_activations[layer_idx + 1][j+3] += net->weights[layer_idx][input_size][j+3];
    }
    
    // Handle remaining bias terms
    for (int j = output_blocks * 4; j < output_size; j++) {
      net->pre_activations[layer_idx + 1][j] += net->weights[layer_idx][input_size][j];
    }
    #endif

    // Special handling for softmax in the output layer
    if (layer_idx == net->size - 2 && net->layers[layer_idx + 1].activation == softmax_single)
    {
      // Find max for numerical stability
      float max_activation = net->pre_activations[layer_idx + 1][0];
      for (int i = 1; i < output_size; i++)
      {
        if (net->pre_activations[layer_idx + 1][i] > max_activation)
          max_activation = net->pre_activations[layer_idx + 1][i];
      }

      // Calculate exp(x - max) and sum
      float exp_sum = 0.0f;
      for (int i = 0; i < output_size; i++)
      {
        net->activations[layer_idx + 1][i] = expf(net->pre_activations[layer_idx + 1][i] - max_activation);
        exp_sum += net->activations[layer_idx + 1][i];
      }

      // Normalize - inverse multiply is faster than division
      float inv_sum = 1.0f / exp_sum;
      for (int i = 0; i < output_size; i++)
      {
        net->activations[layer_idx + 1][i] *= inv_sum;
      }
    }
    else
    {
      // Regular activation for other layers
      for (int i = 0; i < output_size; i++)
      {
        net->activations[layer_idx + 1][i] = 
            net->layers[layer_idx + 1].activation(net->pre_activations[layer_idx + 1][i]);
      }
    }
  }

  return net->activations[net->size - 1];
}

/**
 * @brief Get network prediction for an input (maintains get_result compatibility)
 * @param net Network structure
 * @param input Input values
 * @return Pointer to output layer activations that should NOT be freed
 */
static inline float *get_result(struct Network net, float input[])
{
  // Allocate memory for the result
  float *result = (float *)malloc(net.layers[net.size - 1].length * sizeof(float));
  
  // Forward propagate and copy the output
  float *output = forward_propagate(&net, input);
  memcpy(result, output, net.layers[net.size - 1].length * sizeof(float));
  
  return result;
}

/**
 * @brief Frees all dynamically allocated memory in the network
 * @param net Pointer to the network structure
 */
void freeNetwork(struct Network *net)
{
  // Free weights
  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer].length + 1; i++)
    {
      #ifdef USE_SIMD
      _mm_free(net->weights[layer][i]);
      #else
      free(net->weights[layer][i]);
      #endif
    }
    free(net->weights[layer]);
  }
  free(net->weights);
  
  // Free cached arrays
  for (int layer = 0; layer < net->size; layer++)
  {
    #ifdef USE_SIMD
    _mm_free(net->activations[layer]);
    _mm_free(net->errors[layer]);
    _mm_free(net->pre_activations[layer]);
    #else
    free(net->activations[layer]);
    free(net->errors[layer]);
    free(net->pre_activations[layer]);
    #endif
  }
  free(net->activations);
  free(net->errors);
  free(net->pre_activations);
}

#endif