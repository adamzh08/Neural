#ifndef TRAIN_FUNCTIONS_H
#define TRAIN_FUNCTIONS_H

#include "neural_engine.h"
#include "activation_functions.h"
#include <math.h>
#include <string.h>

/**
 * @struct TrainingParams
 * @brief Configuration parameters for neural network training
 *
 * @member num_samples Total number of training samples
 * @member epochs Number of complete passes through the training data
 * @member learning_rate Initial learning rate for gradient descent
 * @member learning_rate_decay Rate at which learning rate decreases over time
 * @member print_interval How often to print training progress (in epochs)
 * @member momentum Momentum coefficient for gradient descent optimization
 * @member batch_size Number of samples to process before updating weights
 * @member min_delta Minimum change in loss to qualify as improvement
 * @member early_stop_patience Number of epochs to wait for improvement before stopping
 */
struct TrainingParams
{
  int num_samples;
  int epochs;
  float learning_rate;
  float learning_rate_decay;
  int print_interval;
  float momentum;
  int batch_size;
  float min_delta;
  int early_stop_patience;
};

/**
 * @brief Derivative of the sigmoid activation function
 * @param x Input value (typically the pre-activation value)
 * @return Derivative value for backpropagation
 *
 * Computed as: sigmoid(x) * (1 - sigmoid(x))
 * Used in backpropagation to compute gradients
 */
static inline float sigmoid_derivative(float x)
{
  float s = sigmoid(x);
  return s * (1.0f - s);
}

/**
 * @brief Derivative of the tanh activation function
 * @param x Input value
 * @return Derivative value for backpropagation
 *
 * Computed as: 1 - tanh²(x)
 * Efficient implementation avoiding repeated tanh calls
 */
static inline float tanh_derivative(float x)
{
  float t = tanh_func(x);
  return 1.0f - t * t;
}

/**
 * @brief Derivative of the ReLU activation function
 * @param x Input value
 * @return 1 if x > 0, 0 otherwise
 *
 * Simple step function:
 * - Returns 1 for positive inputs
 * - Returns 0 for negative inputs
 * - Undefined at x=0, conventionally returns 1
 */
static inline float relu_derivative(float x)
{
  return x > 0.0f ? 1.0f : 0.0f;
}

/**
 * @brief Derivative of the Leaky ReLU activation function
 * @param x Input value
 * @param alpha Slope for negative values
 * @return 1 if x > 0, alpha otherwise
 */
static inline float lrelu_derivative(float x, float alpha)
{
  return x > 0.0f ? 1.0f : alpha;
}

/**
 * @brief Derivative of the Parametric ReLU activation function
 * @param x Input value
 * @param alpha Learnable slope parameter
 * @return 1 if x > 0, alpha otherwise
 */
static inline float prelu_derivative(float x, float alpha)
{
  return x > 0.0f ? 1.0f : alpha;
}

/**
 * @brief Derivative of the ELU activation function
 * @param x Input value
 * @param alpha Scale parameter for negative values
 * @return 1 if x ≥ 0, alpha * exp(x) otherwise
 */
static inline float elu_derivative(float x, float alpha)
{
  return x >= 0.0f ? 1.0f : alpha * expf(x);
}

/**
 * @brief Derivative of the GELU activation function
 * @param x Input value
 * @return Derivative value for backpropagation
 *
 * Complex derivative implementation taking into account
 * both the Gaussian CDF and tanh approximation components
 */
static inline float gelu_derivative(float x)
{
  float tanh_term = tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)));
  return 0.5 * (1 + tanh_term) + (0.5 * x * (1 - tanh_term * tanh_term) *
                                  sqrt(2 / M_PI) * (1 + 0.134145 * pow(x, 2)));
}

/**
 * @brief Mean Squared Error loss function
 * @param target_values Target values
 * @param predicted_values Network output values
 * @param output_size Size of output layer
 * @return MSE loss value
 *
 * Characteristics:
 * - Suitable for regression problems
 * - Heavily penalizes large errors
 * - Simple derivative for backpropagation
 */
static inline float error_mse(float *target_values, float *predicted_values, int output_size)
{
  float squared_error_sum = 0.0f;
  float error = 0.0f;

  for (int i = 0; i < output_size; i++)
  {
    error = target_values[i] - predicted_values[i];
    squared_error_sum += error * error;
  }

  return squared_error_sum / output_size;
}

/**
 * @brief Root Mean Squared Error loss function
 * @param target_values Target values
 * @param predicted_values Network output values
 * @param output_size Size of output layer
 * @return RMSE loss value
 *
 * Characteristics:
 * - Same scale as the target values
 * - Useful for evaluating prediction accuracy
 * - More interpretable than MSE
 */
static inline float error_rmse(float *target_values, float *predicted_values, int output_size)
{
  float squared_error_sum = 0.0f;
  float error = 0.0f;

  for (int i = 0; i < output_size; i++)
  {
    error = target_values[i] - predicted_values[i];
    squared_error_sum += error * error;
  }

  return (float)sqrt(squared_error_sum / output_size);
}

/**
 * @brief Cross-Entropy Loss function
 * @param target_probabilities Target probabilities
 * @param predicted_probabilities Network output probabilities
 * @param output_size Size of output layer
 * @return Cross-entropy loss value
 *
 * Characteristics:
 * - Suitable for classification problems
 * - Works well with softmax output
 * - Numerically stable implementation
 * - Uses small epsilon to prevent log(0)
 */
static inline float cross_entropy_loss(float *target_probabilities, float *predicted_probabilities, int output_size)
{
  float total_loss = 0.0f;
  const float epsilon = 1e-7f;

  for (int i = 0; i < output_size; i++)
  {
    float clipped_prediction = fmaxf(fminf(predicted_probabilities[i], 1.0f - epsilon), epsilon);
    total_loss -= target_probabilities[i] * logf(clipped_prediction);
  }

  return total_loss / output_size;
}

/**
 * @brief Mean Absolute Error loss function
 * @param target_values Target values
 * @param predicted_values Network output values
 * @param output_size Size of output layer
 * @return MAE loss value
 *
 * Characteristics:
 * - Less sensitive to outliers than MSE
 * - Linear scale of errors
 * - Useful for robust regression
 */
static inline float mean_absolute_error(float *target_values, float *predicted_values, int output_size)
{
  float absolute_error_sum = 0.0f;

  for (int i = 0; i < output_size; i++)
  {
    absolute_error_sum += fabsf(target_values[i] - predicted_values[i]);
  }

  return absolute_error_sum / output_size;
}

/**
 * @brief Huber Loss function
 * @param target_values Target values
 * @param predicted_values Network output values
 * @param output_size Size of output layer
 * @param delta Threshold for switching between L1 and L2 loss
 * @return Huber loss value
 *
 * Characteristics:
 * - Combines MSE and MAE properties
 * - More robust to outliers than MSE
 * - Differentiable everywhere
 */
static inline float huber_loss(float *target_values, float *predicted_values, int output_size, float delta)
{
  float total_loss = 0.0f;

  for (int i = 0; i < output_size; i++)
  {
    float error = target_values[i] - predicted_values[i];

    if (fabsf(error) <= delta)
    {
      total_loss += 0.5f * error * error;
    }
    else
    {
      total_loss += delta * (fabsf(error) - 0.5f * delta);
    }
  }

  return total_loss / output_size;
}

/**
 * @brief Hinge Loss function
 * @param target_values Target values (-1 or 1)
 * @param predicted_values Network output values
 * @param output_size Size of output layer
 * @return Hinge loss value
 *
 * Characteristics:
 * - Used in SVMs and margin-based classifiers
 * - Expects labels in {-1, 1}
 * - Promotes maximum margin classification
 */
static inline float hinge_loss(float *target_values, float *predicted_values, int output_size)
{
  float total_loss = 0.0f;

  for (int i = 0; i < output_size; i++)
  {
    total_loss += fmaxf(0.0f, 1.0f - target_values[i] * predicted_values[i]);
  }

  return total_loss / output_size;
}

/**
 * @brief Gradient clipping function
 * @param gradient_value Pointer to the gradient value to clip
 * @param clip_threshold Maximum allowed gradient magnitude
 *
 * Prevents exploding gradients by scaling down large
 * gradients while preserving their direction
 */
static inline void clip_gradients(float *gradient_value, float clip_threshold)
{
  float gradient_norm = fabsf(*gradient_value);
  if (gradient_norm > clip_threshold)
  {
    *gradient_value *= clip_threshold / gradient_norm;
  }
}

/**
 * @brief Main training function for the neural network
 * @param net Network structure to train
 * @param params Training parameters
 * @param training_data Input training samples
 * @param labels Target output values
 * @param error Error/loss function to use
 *
 * This function implements:
 * - Mini-batch gradient descent
 * - Momentum optimization
 * - Learning rate decay
 * - Early stopping
 * - Progress monitoring
 * - Gradient clipping
 */
static inline void train_network(struct Network net, struct TrainingParams params,
                                 float **training_data, float **labels,
                                 float (*error)(float *, float *, int))
{
  float **layer_activations = (float **)malloc(net.size * sizeof(float *));
  float **layer_errors = (float **)malloc(net.size * sizeof(float *));
  float ***weight_updates = (float ***)malloc((net.size - 1) * sizeof(float **));
  float ***previous_updates = (float ***)malloc((net.size - 1) * sizeof(float **));
  float best_loss = INFINITY;
  int patience_counter = 0;
  float current_learning_rate = params.learning_rate;

  // Initialize memory for layers and updates
  for (int layer = 0; layer < net.size; layer++)
  {
    layer_activations[layer] = (float *)calloc(net.layers[layer].length, sizeof(float));
    layer_errors[layer] = (float *)calloc(net.layers[layer].length, sizeof(float));

    if (layer < net.size - 1)
    {
      weight_updates[layer] = (float **)malloc((net.layers[layer].length + 1) * sizeof(float *));
      previous_updates[layer] = (float **)malloc((net.layers[layer].length + 1) * sizeof(float *));

      for (int i = 0; i < net.layers[layer].length + 1; i++)
      {
        weight_updates[layer][i] = (float *)calloc(net.layers[layer + 1].length, sizeof(float));
        previous_updates[layer][i] = (float *)calloc(net.layers[layer + 1].length, sizeof(float));
      }
    }
  }

  // Training loop
  for (int epoch = 0; epoch < params.epochs; epoch++)
  {
    float epoch_loss = 0.0f;
    int num_batches = params.num_samples / params.batch_size;

    // Shuffle training data
    for (int i = params.num_samples - 1; i > 0; i--)
    {
      int j = rand() % (i + 1);
      float *temp_data = training_data[i];
      float *temp_label = labels[i];
      training_data[i] = training_data[j];
      labels[i] = labels[j];
      training_data[j] = temp_data;
      labels[j] = temp_label;
    }

    // Process mini-batches
    for (int batch = 0; batch < num_batches; batch++)
    {
      // Reset weight updates
      for (int layer = 0; layer < net.size - 1; layer++)
      {
        for (int i = 0; i < net.layers[layer].length + 1; i++)
        {
          memset(weight_updates[layer][i], 0, net.layers[layer + 1].length * sizeof(float));
        }
      }

      float batch_loss = 0.0f;

      // Process each sample in the batch
      for (int sample = 0; sample < params.batch_size; sample++)
      {
        int sample_idx = batch * params.batch_size + sample;

        // Forward pass
        memcpy(layer_activations[0], training_data[sample_idx], net.layers[0].length * sizeof(float));

        for (int layer = 0; layer < net.size - 1; layer++)
        {
          for (int j = 0; j < net.layers[layer + 1].length; j++)
          {
            float sum = 0.0f;
            for (int i = 0; i < net.layers[layer].length; i++)
            {
              sum += layer_activations[layer][i] * net.weights[layer][i][j];
            }
            sum += net.weights[layer][net.layers[layer].length][j]; // Bias

            if (layer == net.size - 2 && net.layers[layer + 1].activation == softmax_single)
            {
              layer_activations[layer + 1][j] = sum; // Pre-activation for softmax
            }
            else
            {
              layer_activations[layer + 1][j] = net.layers[layer + 1].activation(sum);
            }
          }

          // Apply softmax if it's the output layer
          if (layer == net.size - 2 && net.layers[layer + 1].activation == softmax_single)
          {
            float max_val = layer_activations[layer + 1][0];
            for (int i = 1; i < net.layers[layer + 1].length; i++)
            {
              if (layer_activations[layer + 1][i] > max_val)
                max_val = layer_activations[layer + 1][i];
            }

            float sum = 0.0f;
            for (int i = 0; i < net.layers[layer + 1].length; i++)
            {
              layer_activations[layer + 1][i] = expf(layer_activations[layer + 1][i] - max_val);
              sum += layer_activations[layer + 1][i];
            }

            for (int i = 0; i < net.layers[layer + 1].length; i++)
            {
              layer_activations[layer + 1][i] /= sum;
            }
          }
        }

        // Calculate loss
        batch_loss += error(labels[sample_idx], layer_activations[net.size - 1], net.layers[net.size - 1].length);

        // Backward pass
        // Output layer error
        for (int i = 0; i < net.layers[net.size - 1].length; i++)
        {
          layer_errors[net.size - 1][i] = labels[sample_idx][i] - layer_activations[net.size - 1][i];
        }

        // Hidden layers error
        for (int layer = net.size - 2; layer >= 0; layer--)
        {
          for (int i = 0; i < net.layers[layer].length; i++)
          {
            float error_sum = 0.0f;
            for (int j = 0; j < net.layers[layer + 1].length; j++)
            {
              error_sum += layer_errors[layer + 1][j] * net.weights[layer][i][j];
            }
            layer_errors[layer][i] = error_sum;

            if (layer > 0)
            { // Skip input layer activation derivative
              layer_errors[layer][i] *= net.layers[layer].activation_derivative(layer_activations[layer][i]);
            }
          }
        }

        // Accumulate weight updates
        for (int layer = 0; layer < net.size - 1; layer++)
        {
          for (int i = 0; i < net.layers[layer].length; i++)
          {
            for (int j = 0; j < net.layers[layer + 1].length; j++)
            {
              float update = current_learning_rate * layer_errors[layer + 1][j] * layer_activations[layer][i];
              clip_gradients(&update, 5.0f); // Prevent exploding gradients
              weight_updates[layer][i][j] += update;
            }
          }
          // Bias updates
          for (int j = 0; j < net.layers[layer + 1].length; j++)
          {
            float update = current_learning_rate * layer_errors[layer + 1][j];
            clip_gradients(&update, 5.0f);
            weight_updates[layer][net.layers[layer].length][j] += update;
          }
        }
      }

      // Apply accumulated updates with momentum
      for (int layer = 0; layer < net.size - 1; layer++)
      {
        for (int i = 0; i < net.layers[layer].length + 1; i++)
        {
          for (int j = 0; j < net.layers[layer + 1].length; j++)
          {
            float update = weight_updates[layer][i][j] / params.batch_size +
                           params.momentum * previous_updates[layer][i][j];
            net.weights[layer][i][j] += update;
            previous_updates[layer][i][j] = update;
          }
        }
      }

      batch_loss /= params.batch_size;
      epoch_loss += batch_loss;
    }

    epoch_loss /= num_batches;

    // Print progress
    if ((epoch + 1) % params.print_interval == 0)
    {
      printf("Epoch %d/%d - Loss: %.6f - Learning Rate: %.6f\n",
             epoch + 1, params.epochs, epoch_loss, current_learning_rate);
    }

    // Early stopping check
    if (epoch_loss < best_loss - params.min_delta)
    {
      best_loss = epoch_loss;
      patience_counter = 0;
    }
    else
    {
      patience_counter++;
      if (patience_counter >= params.early_stop_patience)
      {
        printf("Early stopping triggered at epoch %d\n", epoch + 1);
        break;
      }
    }

    // Learning rate decay
    current_learning_rate *= params.learning_rate_decay;
  }

  // Free allocated memory
  for (int layer = 0; layer < net.size; layer++)
  {
    free(layer_activations[layer]);
    free(layer_errors[layer]);

    if (layer < net.size - 1)
    {
      for (int i = 0; i < net.layers[layer].length + 1; i++)
      {
        free(weight_updates[layer][i]);
        free(previous_updates[layer][i]);
      }
      free(weight_updates[layer]);
      free(previous_updates[layer]);
    }
  }
  free(layer_activations);
  free(layer_errors);
  free(weight_updates);
  free(previous_updates);
}

/**
 * @brief Tests the network performance on test data
 * @param net Network to test
 * @param test_data Array of test inputs
 * @param test_labels Array of test labels
 * @param num_samples Number of test samples
 * @return Classification accuracy (0.0 to 1.0)
 */
static inline float test_network(struct Network net, float **test_data, float **test_labels, int num_samples)
{
  int correct = 0;
  float *result;

  for (int i = 0; i < num_samples; i++)
  {
    result = get_result(net, test_data[i]);

    // Find predicted and actual class
    int predicted_class = 0;
    int actual_class = 0;
    float max_prob = result[0];

    for (int j = 1; j < net.layers[net.size - 1].length; j++)
    {
      if (result[j] > max_prob)
      {
        max_prob = result[j];
        predicted_class = j;
      }
      if (test_labels[i][j] > test_labels[i][actual_class])
      {
        actual_class = j;
      }
    }

    if (predicted_class == actual_class)
    {
      correct++;
    }

    free(result);
  }

  return (float)correct / num_samples;
}

#endif