#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ANSI escape codes for colors
#define RESET   "\x1b[0m"
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[37m"

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
 *                - Second dimension: output neuron index
 *                - Third dimension: input neuron index (including bias)
 */
struct Network
{
  struct Layers *layers;
  int size;
  float ***weights;
};

/**
 * @brief Creates and initializes a new neural network
 * @param layers Array of layer configurations
 * @param length Number of layers in the network
 * @return Initialized Network structure
 *
 * This function:
 * 1. Creates a new network structure
 * 2. Allocates memory for weights between layers
 * 3. Includes space for bias weights in each layer
 */
struct Network createNetwork(struct Layers layers[], int length)
{
  struct Network net;

  net.layers = layers;
  net.size = length;

  // Allocate memory for layers
  net.weights = (float ***)malloc((length - 1) * sizeof(float **));

  for (int layer = 0; layer < length - 1; layer++)
  {
    net.weights[layer] = (float **)malloc((layers[layer + 1].length) * sizeof(float *));

    for (int input_neuron = 0; input_neuron < layers[layer + 1].length; input_neuron++)
    {
      net.weights[layer][input_neuron] = (float *)malloc((layers[layer].length + 1) * sizeof(float));
    }
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
  FILE *file = fopen(filename, "r");
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
 *
 * Writes all weights including bias weights to a binary file
 * for later restoration
 */
int saveWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer + 1].length; i++)
    {
      fwrite(net->weights[layer][i], sizeof(float), net->layers[layer].length + 1, file);
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
 *
 * Reads previously saved weights from a binary file
 * and restores them into the network
 */
int loadWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (!file)
    return 0;

  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer + 1].length; i++)
    {
      fread(net->weights[layer][i], sizeof(float), net->layers[layer].length + 1, file);
    }
  }
  fclose(file);
  return 1;
}

/**
 * @brief Initializes network weights with random values using Xavier/Glorot initialization
 * @param net Pointer to the network structure
 *
 * Implements Xavier/Glorot initialization which helps with:
 * 1. Preventing vanishing/exploding gradients
 * 2. Maintaining appropriate scale of gradients through the network
 * Scale factor is calculated as sqrt(2 / (fan_in + fan_out))
 */
void randomizeWeights(struct Network *net)
{
  srand(time(NULL));
  for (int layer = 0; layer < net->size - 1; layer++)
  {
    // Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (net->layers[layer].length + net->layers[layer + 1].length));

    for (int i = 0; i < net->layers[layer + 1].length; i++)
    {
      for (int j = 0; j < net->layers[layer].length + 1; j++)
      {
        float r = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        net->weights[layer][i][j] = r * scale;
      }
    }
  }
}

/**
 * @brief Performs forward propagation through the network
 * @param net Network structure
 * @param input Array of input values
 * @return Pointer to array containing output layer activations
 *
 * This function:
 * 1. Propagates input through each layer
 * 2. Applies weights and biases
 * 3. Handles special case for softmax in output layer
 * 4. Applies activation functions
 * 5. Returns final layer output
 *
 * Note: Caller is responsible for freeing returned array
 */
static inline float *get_result(struct Network net, float input[])
{
  float *current_layer_activations = (float *)malloc(net.layers[0].length * sizeof(float));
  memcpy(current_layer_activations, input, net.layers[0].length * sizeof(float));

  float *next_layer_activations = NULL;

  for (int layer_idx = 0; layer_idx < net.size - 1; layer_idx++)
  {
    next_layer_activations = (float *)calloc(net.layers[layer_idx + 1].length, sizeof(float));

    // Forward propagation
    for (int output_neuron = 0; output_neuron < net.layers[layer_idx + 1].length; output_neuron++)
    {
      for (int input_neuron = 0; input_neuron < net.layers[layer_idx].length; input_neuron++)
      {
        next_layer_activations[output_neuron] += current_layer_activations[input_neuron] *
                                                 net.weights[layer_idx][output_neuron][input_neuron];
      }

      // Add bias terms
      next_layer_activations[output_neuron] += net.weights[layer_idx][output_neuron][net.layers[layer_idx].length];
    }

    // Special handling for softmax in the output layer
    if (layer_idx == net.size - 2 && net.layers[layer_idx + 1].activation == softmax_single)
    {
      // Find max for numerical stability
      float max_activation = next_layer_activations[0];
      for (int i = 1; i < net.layers[layer_idx + 1].length; i++)
      {
        if (next_layer_activations[i] > max_activation)
          max_activation = next_layer_activations[i];
      }

      // Calculate exp(x - max) and sum
      float exp_sum = 0.0f;
      for (int i = 0; i < net.layers[layer_idx + 1].length; i++)
      {
        next_layer_activations[i] = expf(next_layer_activations[i] - max_activation);
        exp_sum += next_layer_activations[i];
      }

      // Normalize
      for (int i = 0; i < net.layers[layer_idx + 1].length; i++)
      {
        next_layer_activations[i] /= exp_sum;
      }
    }
    else
    {
      // Regular activation for other layers
      for (int output_neuron = 0; output_neuron < net.layers[layer_idx + 1].length; output_neuron++)
      {
        next_layer_activations[output_neuron] =
            net.layers[layer_idx + 1].activation(next_layer_activations[output_neuron]);
      }
    }

    // Swap buffers
    free(current_layer_activations);
    current_layer_activations = next_layer_activations;
    next_layer_activations = NULL;
  }

  return current_layer_activations;
}

/**
 * @brief Frees all dynamically allocated memory in the network
 * @param net Pointer to the network structure
 *
 * Properly deallocates:
 * 1. Weight arrays for each layer
 * 2. Weight matrix for each input neuron
 * 3. Main weights array
 */
void freeNetwork(struct Network *net)
{
  for (int layer = 0; layer < net->size - 1; layer++)
  {
    for (int i = 0; i < net->layers[layer + 1].length; i++)
    {
      free(net->weights[layer][i]);
    }
    free(net->weights[layer]);
  }
  free(net->weights);
}

#endif