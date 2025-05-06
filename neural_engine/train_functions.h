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
// Vectorized MSE loss
static inline float error_mse(float *t, float *p, int n) {
  float sum = 0.0f;
#ifdef USE_SIMD
  int n4 = n & ~3;
  __m128 acc = _mm_setzero_ps();
  for (int i = 0; i < n4; i += 4) {
      __m128 tv = _mm_loadu_ps(&t[i]);
      __m128 pv = _mm_loadu_ps(&p[i]);
      __m128 diff = _mm_sub_ps(tv, pv);
      acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
  }
  float buf[4]; _mm_storeu_ps(buf, acc);
  sum = buf[0] + buf[1] + buf[2] + buf[3];
  for (int i = n4; i < n; i++) {
      float e = t[i] - p[i]; sum += e*e;
  }
#else
  for (int i = 0; i < n; i++) {
      float e = t[i] - p[i]; sum += e*e;
  }
#endif
  return sum / n;
}

// Vectorized MSE derivative calculation
static inline void mse_derivative(float *t, float *p, float *out, int n) {
#ifdef USE_SIMD
    int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m128 tv = _mm_loadu_ps(&t[i]);
        __m128 pv = _mm_loadu_ps(&p[i]);
        __m128 diff = _mm_sub_ps(tv, pv);
        _mm_storeu_ps(&out[i], diff);
    }
    for (int i = n4; i < n; i++) {
        out[i] = t[i] - p[i];
    }
#else
    for (int i = 0; i < n; i++) {
        out[i] = t[i] - p[i];
    }
#endif
}

// Optimized RMSE calculation with SIMD
static inline float error_rmse(float *target_values, float *predicted_values, int output_size)
{
#ifdef USE_SIMD
    int simd_size = output_size & ~3;
    __m128 sum_vec = _mm_setzero_ps();

    for (int i = 0; i < simd_size; i += 4) {
        __m128 tv = _mm_loadu_ps(&target_values[i]);
        __m128 pv = _mm_loadu_ps(&predicted_values[i]);
        __m128 diff = _mm_sub_ps(tv, pv);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(diff, diff));
    }

    float squared_error_sum = 0.0f;
    float temp[4];
    _mm_storeu_ps(temp, sum_vec);
    squared_error_sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Handle remaining elements
    for (int i = simd_size; i < output_size; i++) {
        float error = target_values[i] - predicted_values[i];
        squared_error_sum += error * error;
    }

#else
    float squared_error_sum = 0.0f;
    int i = 0;
    int unroll_size = output_size & ~3;
    
    // Loop unrolling for better pipelining
    for (; i < unroll_size; i += 4) {
        float error1 = target_values[i] - predicted_values[i];
        float error2 = target_values[i+1] - predicted_values[i+1];
        float error3 = target_values[i+2] - predicted_values[i+2];
        float error4 = target_values[i+3] - predicted_values[i+3];
        squared_error_sum += error1 * error1 + error2 * error2 + error3 * error3 + error4 * error4;
    }
    
    for (; i < output_size; i++) {
        float error = target_values[i] - predicted_values[i];
        squared_error_sum += error * error;
    }
#endif

    return (float)sqrt(squared_error_sum / output_size);
}

// Optimized cross-entropy loss with SIMD and numerical stability
static inline float cross_entropy_loss(float *t, float *p, int n) {
    const float eps = 1e-7f;
    float loss = 0.0f;
    
#ifdef USE_SIMD
    int n4 = n & ~3;
    __m128 eps_vec = _mm_set1_ps(eps);
    __m128 one_minus_eps = _mm_set1_ps(1.0f - eps);
    __m128 sum = _mm_setzero_ps();
    
    for (int i = 0; i < n4; i += 4) {
        __m128 tv = _mm_loadu_ps(&t[i]);
        __m128 pv = _mm_loadu_ps(&p[i]);
        
        // Clamp probabilities
        pv = _mm_min_ps(_mm_max_ps(pv, eps_vec), one_minus_eps);
        
        // Calculate log
        __m128 log_p;
        float temp[4];
        _mm_storeu_ps(temp, pv);
        temp[0] = logf(temp[0]);
        temp[1] = logf(temp[1]);
        temp[2] = logf(temp[2]);
        temp[3] = logf(temp[3]);
        log_p = _mm_loadu_ps(temp);
        
        // Multiply by target and accumulate
        sum = _mm_add_ps(sum, _mm_mul_ps(tv, log_p));
    }
    
    // Reduce sum vector
    float temp[4];
    _mm_storeu_ps(temp, sum);
    loss = -(temp[0] + temp[1] + temp[2] + temp[3]);
    
    // Handle remaining elements
    for (int i = n4; i < n; i++) {
        float pp = fmaxf(fminf(p[i], 1.0f-eps), eps);
        loss -= t[i] * logf(pp);
    }
#else
    int i = 0;
    int unroll_size = n & ~3;
    
    for (; i < unroll_size; i += 4) {
        float pp1 = fmaxf(fminf(p[i], 1.0f-eps), eps);
        float pp2 = fmaxf(fminf(p[i+1], 1.0f-eps), eps);
        float pp3 = fmaxf(fminf(p[i+2], 1.0f-eps), eps);
        float pp4 = fmaxf(fminf(p[i+3], 1.0f-eps), eps);
        loss -= t[i] * logf(pp1) + t[i+1] * logf(pp2) + 
                t[i+2] * logf(pp3) + t[i+3] * logf(pp4);
    }
    
    for (; i < n; i++) {
        float pp = fmaxf(fminf(p[i], 1.0f-eps), eps);
        loss -= t[i] * logf(pp);
    }
#endif

    return loss / n;
}

// Optimized mean absolute error with SIMD
static inline float mean_absolute_error(float *target_values, float *predicted_values, int output_size)
{
#ifdef USE_SIMD
    int simd_size = output_size & ~3;
    __m128 sum_vec = _mm_setzero_ps();
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 tv = _mm_loadu_ps(&target_values[i]);
        __m128 pv = _mm_loadu_ps(&predicted_values[i]);
        __m128 diff = _mm_sub_ps(tv, pv);
        // Absolute value using bit manipulation
        __m128 abs_mask = _mm_set1_ps(-0.0f);
        __m128 abs_diff = _mm_andnot_ps(abs_mask, diff);
        sum_vec = _mm_add_ps(sum_vec, abs_diff);
    }
    
    float absolute_error_sum = 0.0f;
    float temp[4];
    _mm_storeu_ps(temp, sum_vec);
    absolute_error_sum = temp[0] + temp[1] + temp[2] + temp[3];
    
    for (int i = simd_size; i < output_size; i++) {
        absolute_error_sum += fabsf(target_values[i] - predicted_values[i]);
    }
#else
    float absolute_error_sum = 0.0f;
    int i = 0;
    int unroll_size = output_size & ~3;
    
    for (; i < unroll_size; i += 4) {
        absolute_error_sum += fabsf(target_values[i] - predicted_values[i]) +
                            fabsf(target_values[i+1] - predicted_values[i+1]) +
                            fabsf(target_values[i+2] - predicted_values[i+2]) +
                            fabsf(target_values[i+3] - predicted_values[i+3]);
    }
    
    for (; i < output_size; i++) {
        absolute_error_sum += fabsf(target_values[i] - predicted_values[i]);
    }
#endif

    return absolute_error_sum / output_size;
}

// Optimized Huber loss with SIMD
static inline float huber_loss(float *target_values, float *predicted_values, int output_size, float delta)
{
#ifdef USE_SIMD
    int simd_size = output_size & ~3;
    __m128 sum_vec = _mm_setzero_ps();
    __m128 delta_vec = _mm_set1_ps(delta);
    __m128 half_vec = _mm_set1_ps(0.5f);
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 tv = _mm_loadu_ps(&target_values[i]);
        __m128 pv = _mm_loadu_ps(&predicted_values[i]);
        __m128 diff = _mm_sub_ps(tv, pv);
        
        // Calculate absolute difference
        __m128 abs_mask = _mm_set1_ps(-0.0f);
        __m128 abs_diff = _mm_andnot_ps(abs_mask, diff);
        
        // Create mask for |diff| <= delta
        __m128 mask = _mm_cmple_ps(abs_diff, delta_vec);
        
        // Calculate quadratic term: 0.5 * diff^2
        __m128 quad = _mm_mul_ps(half_vec, _mm_mul_ps(diff, diff));
        
        // Calculate linear term: delta * (|diff| - 0.5 * delta)
        __m128 lin = _mm_mul_ps(delta_vec, 
                               _mm_sub_ps(abs_diff, 
                                        _mm_mul_ps(half_vec, delta_vec)));
        
        // Select based on mask
        __m128 result = _mm_or_ps(_mm_and_ps(mask, quad),
                                 _mm_andnot_ps(mask, lin));
        
        sum_vec = _mm_add_ps(sum_vec, result);
    }
    
    float total_loss = 0.0f;
    float temp[4];
    _mm_storeu_ps(temp, sum_vec);
    total_loss = temp[0] + temp[1] + temp[2] + temp[3];
    
    for (int i = simd_size; i < output_size; i++) {
        float error = target_values[i] - predicted_values[i];
        if (fabsf(error) <= delta) {
            total_loss += 0.5f * error * error;
        } else {
            total_loss += delta * (fabsf(error) - 0.5f * delta);
        }
    }
#else
    float total_loss = 0.0f;
    int i = 0;
    int unroll_size = output_size & ~3;
    
    for (; i < unroll_size; i += 4) {
        for (int j = 0; j < 4; j++) {
            float error = target_values[i+j] - predicted_values[i+j];
            if (fabsf(error) <= delta) {
                total_loss += 0.5f * error * error;
            } else {
                total_loss += delta * (fabsf(error) - 0.5f * delta);
            }
        }
    }
    
    for (; i < output_size; i++) {
        float error = target_values[i] - predicted_values[i];
        if (fabsf(error) <= delta) {
            total_loss += 0.5f * error * error;
        } else {
            total_loss += delta * (fabsf(error) - 0.5f * delta);
        }
    }
#endif

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
static inline void clip_gradients(float *g, float thr) {
  #ifdef USE_SIMD
      __m128 gv = _mm_set1_ps(*g);
      __m128 tv = _mm_set1_ps(thr);
      __m128 nm = _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), gv), tv);
      float v = *g;
      if (fabsf(v) > thr) *g = v * (thr / fabsf(v));
  #else
      float val = *g;
      if (fabsf(val) > thr) *g = val * (thr / fabsf(val));
  #endif
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
  // Allocate and align memory for intermediate results
  int max_layer_size = 0;
  for (int i = 0; i < net.size; i++) {
    if (net.layers[i].length > max_layer_size) max_layer_size = net.layers[i].length;
  }

  // Aligned memory allocation for better SIMD performance
  float **weight_updates;
  float **previous_updates;
  
  #ifdef USE_SIMD
  weight_updates = (float **)_mm_malloc((net.size - 1) * sizeof(float *), 16);
  previous_updates = (float **)_mm_malloc((net.size - 1) * sizeof(float *), 16);
  #else
  weight_updates = (float **)malloc((net.size - 1) * sizeof(float *));
  previous_updates = (float **)malloc((net.size - 1) * sizeof(float *));
  #endif

  // Initialize memory pools for updates
  for (int layer = 0; layer < net.size - 1; layer++) {
    int weights_size = (net.layers[layer].length + 1) * net.layers[layer + 1].length;
    #ifdef USE_SIMD
    weight_updates[layer] = (float *)_mm_malloc(weights_size * sizeof(float), 16);
    previous_updates[layer] = (float *)_mm_malloc(weights_size * sizeof(float), 16);
    #else
    weight_updates[layer] = (float *)malloc(weights_size * sizeof(float));
    previous_updates[layer] = (float *)malloc(weights_size * sizeof(float));
    #endif
    memset(previous_updates[layer], 0, weights_size * sizeof(float));
  }

  float best_loss = INFINITY;
  int patience_counter = 0;
  float current_learning_rate = params.learning_rate;

  // Pre-compute batch indices for random access - Fix infinite loop
  int *batch_indices = (int *)malloc(params.num_samples * sizeof(int));
  for (int i = 0; i < params.num_samples; i++) {  // Fixed initialization loop
    batch_indices[i] = i;
  }

  // Training loop with early stopping
  #ifdef USE_SIMD
  __m128 learning_rate_vec, momentum_vec;
  #endif
  
  for (int epoch = 0; epoch < params.epochs && patience_counter < params.early_stop_patience; epoch++) {
    float epoch_loss = 0.0f;
    int num_batches = params.num_samples / params.batch_size;

    // Efficient Fisher-Yates shuffle with progress tracking
    /*
    if ((epoch + 1) % params.print_interval == 0) {
      printf("Shuffling training data for epoch %d...\n", epoch + 1);
    }
    */
    
    for (int i = params.num_samples - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      int temp = batch_indices[i];
      batch_indices[i] = batch_indices[j];
      batch_indices[j] = temp;
    }

    /*
    if ((epoch + 1) % params.print_interval == 0) {
      printf("Processing batches...\n");
    }
    */

    // Process mini-batches with vectorized operations
    #ifdef USE_SIMD
    learning_rate_vec = _mm_set1_ps(current_learning_rate);
    momentum_vec = _mm_set1_ps(params.momentum);
    #endif

    #pragma omp parallel for reduction(+:epoch_loss) if(num_batches > 4)
    for (int batch = 0; batch < num_batches; batch++) {
      float batch_loss = 0.0f;

      // Zero out weight updates for this batch
      for (int layer = 0; layer < net.size - 1; layer++) {
        int weights_size = (net.layers[layer].length + 1) * net.layers[layer + 1].length;
        memset(weight_updates[layer], 0, weights_size * sizeof(float));
      }

      // Process each sample in the batch
      for (int sample = 0; sample < params.batch_size; sample++) {
        int sample_idx = batch_indices[batch * params.batch_size + sample];
        if (sample_idx >= params.num_samples) continue;  // Safety check
        
        // Forward propagation
        float *output = forward_propagate(&net, training_data[sample_idx]);
        
        // Calculate loss
        batch_loss += error(labels[sample_idx], output, net.layers[net.size - 1].length);

        // Backward pass
        // Calculate output layer error (derivative of loss function)
        const int output_size = net.layers[net.size - 1].length;
        for (int i = 0; i < output_size; i++) {
          net.errors[net.size - 1][i] = output[i] - labels[sample_idx][i];
        }

        // Backward propagate errors
        for (int layer = net.size - 2; layer >= 0; layer--) {
          const int current_layer_size = net.layers[layer].length;
          const int next_layer_size = net.layers[layer + 1].length;

          #ifdef USE_SIMD
          int simd_size = current_layer_size & ~3;
          
          for (int i = 0; i < simd_size; i += 4) {
            __m128 error_sum = _mm_setzero_ps();
            
            for (int j = 0; j < next_layer_size; j++) {
              __m128 weights = _mm_set_ps(
                net.weights[layer][i+3][j],
                net.weights[layer][i+2][j],
                net.weights[layer][i+1][j],
                net.weights[layer][i][j]
              );
              __m128 next_error = _mm_set1_ps(net.errors[layer + 1][j]);
              error_sum = _mm_add_ps(error_sum, _mm_mul_ps(weights, next_error));
            }
            
            float temp[4];
            _mm_storeu_ps(temp, error_sum);
            
            for (int k = 0; k < 4; k++) {
              if (layer > 0) {
                net.errors[layer][i+k] = temp[k] * 
                  net.layers[layer].activation_derivative(net.activations[layer][i+k]);
              } else {
                net.errors[layer][i+k] = temp[k];
              }
            }
          }
          
          // Handle remaining neurons
          for (int i = simd_size; i < current_layer_size; i++) {
            float error_sum = 0.0f;
            for (int j = 0; j < next_layer_size; j++) {
              error_sum += net.weights[layer][i][j] * net.errors[layer + 1][j];
            }
            if (layer > 0) {
              net.errors[layer][i] = error_sum * 
                net.layers[layer].activation_derivative(net.activations[layer][i]);
            } else {
              net.errors[layer][i] = error_sum;
            }
          }
          #else
          // Non-SIMD backward propagation with loop unrolling
          for (int i = 0; i < current_layer_size; i++) {
            float error_sum = 0.0f;
            int j = 0;
            int unroll_size = next_layer_size & ~3;
            
            for (; j < unroll_size; j += 4) {
              error_sum += net.weights[layer][i][j] * net.errors[layer + 1][j]
                        + net.weights[layer][i][j+1] * net.errors[layer + 1][j+1]
                        + net.weights[layer][i][j+2] * net.errors[layer + 1][j+2]
                        + net.weights[layer][i][j+3] * net.errors[layer + 1][j+3];
            }
            
            for (; j < next_layer_size; j++) {
              error_sum += net.weights[layer][i][j] * net.errors[layer + 1][j];
            }
            
            if (layer > 0) {
              net.errors[layer][i] = error_sum * 
                net.layers[layer].activation_derivative(net.activations[layer][i]);
            } else {
              net.errors[layer][i] = error_sum;
            }
          }
          #endif

          // Update weights
          const float scaled_lr = -current_learning_rate / params.batch_size;
          #ifdef USE_SIMD
          __m128 lr_vec = _mm_set1_ps(scaled_lr);
          __m128 momentum_v = _mm_set1_ps(params.momentum);
          
          for (int i = 0; i <= current_layer_size; i++) {
            float activation = (i < current_layer_size) ? net.activations[layer][i] : 1.0f;
            __m128 act_vec = _mm_set1_ps(activation);
            
            int j = 0;
            int simd_size = next_layer_size & ~3;
            
            for (; j < simd_size; j += 4) {
              __m128 error_vec = _mm_loadu_ps(&net.errors[layer + 1][j]);
              __m128 delta = _mm_mul_ps(_mm_mul_ps(lr_vec, error_vec), act_vec);
              
              int idx = i * next_layer_size + j;
              __m128 prev_update = _mm_loadu_ps(&previous_updates[layer][idx]);
              __m128 momentum_update = _mm_mul_ps(momentum_v, prev_update);
              __m128 final_update = _mm_add_ps(delta, momentum_update);
              
              // Apply update
              __m128 weights_vec = _mm_loadu_ps(&net.weights[layer][i][j]);
              weights_vec = _mm_add_ps(weights_vec, final_update);
              _mm_storeu_ps(&net.weights[layer][i][j], weights_vec);
              
              // Store for next iteration
              _mm_storeu_ps(&previous_updates[layer][idx], final_update);
            }
            
            // Handle remaining weights
            for (; j < next_layer_size; j++) {
              int idx = i * next_layer_size + j;
              float delta = scaled_lr * net.errors[layer + 1][j] * activation;
              float update = delta + params.momentum * previous_updates[layer][idx];
              net.weights[layer][i][j] += update;
              previous_updates[layer][idx] = update;
            }
          }
          #else
          for (int i = 0; i <= current_layer_size; i++) {
            float activation = (i < current_layer_size) ? net.activations[layer][i] : 1.0f;
            
            int j = 0;
            int unroll_size = next_layer_size & ~3;
            
            for (; j < unroll_size; j += 4) {
              int idx = i * next_layer_size + j;
              
              for (int k = 0; k < 4; k++) {
                float delta = scaled_lr * net.errors[layer + 1][j+k] * activation;
                float update = delta + params.momentum * previous_updates[layer][idx+k];
                net.weights[layer][i][j+k] += update;
                previous_updates[layer][idx+k] = update;
              }
            }
            
            for (; j < next_layer_size; j++) {
              int idx = i * next_layer_size + j;
              float delta = scaled_lr * net.errors[layer + 1][j] * activation;
              float update = delta + params.momentum * previous_updates[layer][idx];
              net.weights[layer][i][j] += update;
              previous_updates[layer][idx] = update;
            }
          }
          #endif
        }
      }

      batch_loss /= params.batch_size;
      epoch_loss += batch_loss;

      /*
      if ((batch + 1) % 100 == 0 && (epoch + 1) % params.print_interval == 0) {
        printf("Batch %d/%d\r", batch + 1, num_batches);
        fflush(stdout);
      }
      */
    }

    epoch_loss /= num_batches;

    // Print progress and check early stopping
    if ((epoch + 1) % params.print_interval == 0) {
      printf("Epoch %d/%d - Loss: %.6f - LR: %.6f\n",
             epoch + 1, params.epochs, epoch_loss, current_learning_rate);
    }

    if (epoch_loss < best_loss - params.min_delta) {
      best_loss = epoch_loss;
      patience_counter = 0;
    } else {
      patience_counter++;
    }

    current_learning_rate *= params.learning_rate_decay;
  }

  // Free allocated memory
  free(batch_indices);
  for (int layer = 0; layer < net.size - 1; layer++) {
    #ifdef USE_SIMD
    _mm_free(weight_updates[layer]);
    _mm_free(previous_updates[layer]);
    #else
    free(weight_updates[layer]);
    free(previous_updates[layer]);
    #endif
  }
  #ifdef USE_SIMD
  _mm_free(weight_updates);
  _mm_free(previous_updates);
  #else
  free(weight_updates);
  free(previous_updates);
  #endif
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