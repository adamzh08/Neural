/**
 * @file main.c
 * @brief MNIST digit recognition using neural network implementation
 *
 * This program implements a neural network to recognize handwritten digits
 * from the MNIST dataset. It includes:
 * - Data loading and preprocessing
 * - Network architecture definition
 * - Training and testing procedures
 * - Weight persistence
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "neural_engine/activation_functions.h"
#include "neural_engine/neural_engine.h"
#include "neural_engine/train_functions.h"
#include "mnist.h"

/** Size of input layer (28x28 pixel images) */
#define INPUT_DATA_SIZE 784

/** Size of output layer (10 digits, 0-9) */
#define OUTPUT_DATA_SIZE 10

/** Number of training samples in MNIST */
#define TRAIN_SAMPLES 60000

/** Number of test samples in MNIST */
#define TEST_SAMPLES 10000

/**
 * @brief Converts a scalar label to one-hot encoded array
 * @param label Input label (0-9)
 * @param output Array to store one-hot encoding
 *
 * Creates a binary vector where only the index
 * corresponding to the label is set to 1.0,
 * all other indices are 0.0
 */
void to_one_hot(int label, float *output)
{
  for (int i = 0; i < OUTPUT_DATA_SIZE; i++)
  {
    output[i] = (i == label) ? 1.0f : 0.0f;
  }
}

/**
 * @brief Determines the predicted digit from network output
 * @param output Network output array (probabilities)
 * @return Index (0-9) with highest probability
 *
 * Returns the index of the maximum value in the output array,
 * which corresponds to the most likely digit prediction
 */
int get_prediction(float *output)
{
  int max_idx = 0;
  float max_val = output[0];
  for (int i = 1; i < OUTPUT_DATA_SIZE; i++)
  {
    if (output[i] > max_val)
    {
      max_val = output[i];
      max_idx = i;
    }
  }
  return max_idx;
}

/**
 * @brief Main program entry point
 * @return 0 on successful execution
 *
 * Program flow:
 * 1. Loads MNIST dataset
 * 2. Preprocesses training data:
 *    - Normalizes pixel values to [0,1]
 *    - Converts labels to one-hot encoding
 * 3. Creates neural network with architecture:
 *    - Input layer: 784 neurons (28x28 pixels)
 *    - Hidden layer: 128 neurons with ReLU activation
 *    - Output layer: 10 neurons with softmax activation
 * 4. Either loads existing weights or initializes new ones
 * 5. Trains the network using mini-batch gradient descent
 * 6. Tests network performance
 * 7. Saves trained weights
 * 8. Cleans up allocated memory
 */
int main(void)
{
  printf("Loading MNIST data...\n");
  if (!load_mnist())
  {
    printf("Failed to load MNIST dataset\n");
    return 1;
  }

  // Prepare training data
  float **input_samples = (float **)malloc(TRAIN_SAMPLES * sizeof(float *));
  float **target_labels = (float **)malloc(TRAIN_SAMPLES * sizeof(float *));
  float **test_inputs = (float **)malloc(TEST_SAMPLES * sizeof(float *));
  float **test_targets = (float **)malloc(TEST_SAMPLES * sizeof(float *));

  // Initialize training data
  for (int i = 0; i < TRAIN_SAMPLES; i++)
  {
    input_samples[i] = (float *)malloc(INPUT_DATA_SIZE * sizeof(float));
    target_labels[i] = (float *)malloc(OUTPUT_DATA_SIZE * sizeof(float));

    // Normalize pixel values to [0,1]
    for (int j = 0; j < INPUT_DATA_SIZE; j++)
    {
      input_samples[i][j] = (float)training_images[i][j] / 255.0f;
    }

    to_one_hot(training_labels[i], target_labels[i]);
  }

  // Initialize test data
  for (int i = 0; i < TEST_SAMPLES; i++)
  {
    test_inputs[i] = (float *)malloc(INPUT_DATA_SIZE * sizeof(float));
    test_targets[i] = (float *)malloc(OUTPUT_DATA_SIZE * sizeof(float));

    // Normalize pixel values to [0,1]
    for (int j = 0; j < INPUT_DATA_SIZE; j++)
    {
      test_inputs[i][j] = (float)test_images[i][j] / 255.0f;
    }

    to_one_hot(test_labels[i], test_targets[i]);
  }

  // Define network architecture
  struct Layers network_architecture[] = {
      {INPUT_DATA_SIZE, NULL, NULL},                         // Input layer: 784 neurons
      {128, gelu, gelu_derivative},                          // Hidden layer: 128 neurons with ReLU
      {32, gelu, gelu_derivative},                          // Hidden layer: 128 neurons with ReLU
      {OUTPUT_DATA_SIZE, softmax_single, softmax_derivative} // Output layer: 10 neurons with softmax
  };

  int num_layers = sizeof(network_architecture) / sizeof(network_architecture[0]);
  struct Network network = createNetwork(network_architecture, num_layers);

  // Load existing weights or initialize new ones
  if (file_exists("network_weights.bin"))
  {
    printf("Loading existing weights...\n");
    loadWeights(&network, "network_weights.bin");
  }
  else
  {
    printf("Initializing new weights...\n");
    randomizeWeights(&network);
  }

  // Configure training parameters
  struct TrainingParams training_config = {
      .num_samples = TRAIN_SAMPLES,
      .epochs = 50,
      .learning_rate = 0.001f,
      .learning_rate_decay = 0.95f,
      .print_interval = 1,
      .batch_size = 32,
      .momentum = 0.9f,
      .min_delta = 0.0001f,
      .early_stop_patience = 5};

  // Start training
  clock_t training_start = clock();
  printf("Starting training...\n");

  train_network(network, training_config, input_samples, target_labels, cross_entropy_loss);

  double training_time = ((double)(clock() - training_start)) / CLOCKS_PER_SEC;
  printf("\nTotal training time: %.2f seconds\n", training_time);

  // Test the network
  printf("\nTesting network performance...\n");
  float accuracy = test_network(network, test_inputs, test_targets, TEST_SAMPLES);
  printf("Test accuracy: %.2f%%\n", accuracy * 100.0f);

  for (int i = 0; i < 20; i++)
  {
    float *res = get_result(network, test_inputs[i]);

    for (int i1 = 0; i1 < 10; i1++)
    {
      if (res[i1] > 0.7)
      {
        printf(RED "%d - %d - %f" RESET "\n", test_labels[i], i1, res[i1]);
      }
      else
      {
        printf("%d - %d - %f\n", test_labels[i], i1, res[i1]);
      }
    }

    free(res);
  }

  // Save the trained weights
  printf("Saving weights...\n");
  if (!saveWeights(&network, "network_weights.bin"))
  {
    printf("Error saving weights\n");
  }

  // Clean up allocated memory
  for (int i = 0; i < TRAIN_SAMPLES; i++)
  {
    free(input_samples[i]);
    free(target_labels[i]);
  }
  for (int i = 0; i < TEST_SAMPLES; i++)
  {
    free(test_inputs[i]);
    free(test_targets[i]);
  }
  free(input_samples);
  free(target_labels);
  free(test_inputs);
  free(test_targets);
  freeNetwork(&network);

  return 0;
}