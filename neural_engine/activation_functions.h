#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <math.h>

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return Output in range (0,1)
 *
 * Characteristics:
 * - Smooth, continuous function
 * - Output range: (0,1)
 * - Commonly used in binary classification
 * - Can cause vanishing gradient problems
 */
static inline float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Hyperbolic tangent activation function
 * @param x Input value
 * @return Output in range (-1,1)
 *
 * Characteristics:
 * - Zero-centered output
 * - Output range: (-1,1)
 * - Stronger gradients than sigmoid
 * - Still can have vanishing gradient issues
 */
static inline float tanh_func(float x)
{
  return tanhf(x);
}

/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * @param x Input value
 * @return max(0,x)
 *
 * Characteristics:
 * - Simple and computationally efficient
 * - No vanishing gradient for positive values
 * - Can cause "dying ReLU" problem
 * - Most commonly used activation in modern networks
 */
static inline float relu(float x)
{
  return x > 0.0f ? x : 0.0f;
}

/**
 * @brief Leaky ReLU activation function
 * @param x Input value
 * @param alpha Slope for negative values (typically small, e.g., 0.01)
 * @return x if x > 0, alpha * x otherwise
 *
 * Characteristics:
 * - Prevents dying ReLU problem
 * - Small gradient for negative values
 * - No vanishing gradient
 */
static inline float lrelu(float x, float alpha)
{
  return x > 0.0f ? x : alpha * x;
}

/**
 * @brief Parametric ReLU activation function
 * @param x Input value
 * @param alpha Learnable parameter for negative values
 * @return x if x > 0, alpha * x otherwise
 *
 * Characteristics:
 * - Similar to Leaky ReLU but with learnable alpha
 * - More flexible than standard ReLU
 * - Requires additional parameter training
 */
static inline float prelu(float x, float alpha)
{
  return x > 0.0f ? x : alpha * x;
}

/**
 * @brief Exponential Linear Unit activation function
 * @param x Input value
 * @param alpha Scale for the negative part
 * @return x if x ≥ 0, alpha * (exp(x) - 1) otherwise
 *
 * Characteristics:
 * - Smooth function including at x=0
 * - Can produce negative values
 * - Better handling of noise
 * - Self-regularizing
 */
static inline float elu(float x, float alpha)
{
  return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

/**
 * @brief Softmax activation function for entire layer
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 *
 * Characteristics:
 * - Converts inputs to probability distribution
 * - Outputs sum to 1.0
 * - Commonly used in classification
 * - Numerically stable implementation
 */
static inline void softmax(float *input, float *output, int size)
{
  float max_val = input[0];
  for (int i = 1; i < size; i++)
  {
    if (input[i] > max_val)
    {
      max_val = input[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }

  for (int i = 0; i < size; i++)
  {
    output[i] /= sum;
  }
}

/**
 * @brief Single-input softmax for network structure
 * @param x Input value
 * @return Exponential of input (partial softmax)
 *
 * Note: This is only part of the softmax calculation.
 * Full normalization happens in the network forward pass.
 */
static inline float softmax_single(float x)
{
  return expf(x);
}

/**
 * @brief Derivative of softmax function
 * @param x Input value
 * @return Derivative value for backpropagation
 */
static inline float softmax_derivative(float x)
{
  float s = softmax_single(x);
  return s * (1 - s);
}

/**
 * @brief Gaussian Error Linear Unit (GELU) activation
 * @param x Input value
 * @return GELU activation value
 *
 * Characteristics:
 * - Smooth approximation of ReLU
 * - Used in modern transformers
 * - Combines properties of dropout and ReLU
 * - More computationally expensive
 */
static inline float gelu(float x)
{
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

#endif // ACTIVATION_FUNCTIONS_H