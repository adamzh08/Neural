#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <math.h>

// Enable SIMD intrinsics based on platform
#if defined(__SSE__) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#define USE_SIMD 1
#endif

// Constants for exponential approximations
#define EXP_A 12102203.0f
#define EXP_B 1064866805.0f
#define EXP_C 12102203.0f
#define SIGMOID_MAX 15.0f
#define SIGMOID_MIN -15.0f

// Custom fast exp approximation
// Can be faster than expf() for activation functions
static inline float fast_exp(float x) {
    x = x > 88.0f ? 88.0f : x;  // Clamp to avoid overflow
    x = x < -88.0f ? -88.0f : x;

    // Use bit manipulation for fast computation
    union {
        float f;
        int i;
    } value;

    // exp(x) ≈ 2^(1.44269504 * x)
    const float log2e = 1.44269504f;
    float y = x * log2e;
    
    // Split into integer and fractional parts
    int integer = (int)y;
    float frac = y - integer;
    
    // Calculate 2^frac using polynomial approximation
    // 2^frac ≈ 1.0f + frac * (0.693147f + frac * 0.2402265f)
    float poly = 1.0f + frac * (0.693147f + frac * 0.2402265f);
    
    // Calculate 2^integer using bit manipulation
    value.i = (integer + 127) << 23;
    
    return value.f * poly;
}

/**
 * @brief Sigmoid activation function with SIMD optimization
 * @param x Input value
 * @return Output in range (0,1)
 */
static inline float sigmoid(float x)
{
    // Clamp input to avoid numerical issues
    x = x > SIGMOID_MAX ? SIGMOID_MAX : x;
    x = x < SIGMOID_MIN ? SIGMOID_MIN : x;
    
    return 1.0f / (1.0f + fast_exp(-x));
}

/**
 * @brief SIMD-optimized sigmoid for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 */
static inline void sigmoid_vector(float *input, float *output, int size)
{
#ifdef USE_SIMD
    // Process 4 elements at once with SSE
    int simd_size = size & ~3; // Round down to multiple of 4
    
    __m128 one = _mm_set1_ps(1.0f);
    __m128 max_val = _mm_set1_ps(SIGMOID_MAX);
    __m128 min_val = _mm_set1_ps(SIGMOID_MIN);
    
    for (int i = 0; i < simd_size; i += 4) {
        // Load 4 input values
        __m128 x = _mm_loadu_ps(&input[i]);
        
        // Clamp values
        x = _mm_min_ps(x, max_val);
        x = _mm_max_ps(x, min_val);
        
        // Negate
        __m128 neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
        
        // Calculate exp(-x) 
        // Either using the fast_exp method for each component
        // or a vectorized approximation
        
        // For this example, we'll use individual calls to fast_exp
        float temp[4];
        _mm_storeu_ps(temp, neg_x);
        
        __m128 exp_result = _mm_set_ps(
            fast_exp(temp[3]),
            fast_exp(temp[2]),
            fast_exp(temp[1]),
            fast_exp(temp[0])
        );
        
        // 1.0 / (1.0 + exp(-x))
        __m128 result = _mm_div_ps(one, _mm_add_ps(one, exp_result));
        
        // Store result
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = sigmoid(input[i]);
    }
#else
    // Scalar fallback with loop unrolling for better pipelining
    int i = 0;
    int unroll_size = size & ~3; // Round down to multiple of 4
    
    for (; i < unroll_size; i += 4) {
        output[i] = sigmoid(input[i]);
        output[i+1] = sigmoid(input[i+1]);
        output[i+2] = sigmoid(input[i+2]);
        output[i+3] = sigmoid(input[i+3]);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = sigmoid(input[i]);
    }
#endif
}

/**
 * @brief Hyperbolic tangent activation function
 * @param x Input value
 * @return Output in range (-1,1)
 */
static inline float tanh_func(float x)
{
    // Fast approximation of tanh using sigmoid
    // tanh(x) = 2*sigmoid(2x) - 1
    float sig2x = sigmoid(2.0f * x);
    return 2.0f * sig2x - 1.0f;
}

/**
 * @brief SIMD-optimized tanh for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 */
static inline void tanh_vector(float *input, float *output, int size)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    
    __m128 two = _mm_set1_ps(2.0f);
    __m128 one = _mm_set1_ps(1.0f);
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 x2 = _mm_mul_ps(two, x);
        
        // Calculate sigmoid(2x) for each component
        float temp[4];
        _mm_storeu_ps(temp, x2);
        
        __m128 sig_result = _mm_set_ps(
            sigmoid(temp[3]),
            sigmoid(temp[2]),
            sigmoid(temp[1]),
            sigmoid(temp[0])
        );
        
        // 2*sigmoid(2x) - 1
        __m128 result = _mm_sub_ps(_mm_mul_ps(two, sig_result), one);
        
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = tanh_func(input[i]);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = tanh_func(input[i]);
        output[i+1] = tanh_func(input[i+1]);
        output[i+2] = tanh_func(input[i+2]);
        output[i+3] = tanh_func(input[i+3]);
    }
    
    for (; i < size; i++) {
        output[i] = tanh_func(input[i]);
    }
#endif
}

/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * @param x Input value
 * @return max(0,x)
 */
static inline float relu(float x)
{
    // Simple comparison is faster than branching
    return x * (x > 0.0f);
}

/**
 * @brief SIMD-optimized ReLU for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 */
static inline void relu_vector(float *input, float *output, int size)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    __m128 zero = _mm_setzero_ps();
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 result = _mm_max_ps(x, zero);
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = relu(input[i]);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = relu(input[i]);
        output[i+1] = relu(input[i+1]);
        output[i+2] = relu(input[i+2]);
        output[i+3] = relu(input[i+3]);
    }
    
    for (; i < size; i++) {
        output[i] = relu(input[i]);
    }
#endif
}

/**
 * @brief Leaky ReLU activation function
 * @param x Input value
 * @param alpha Slope for negative values (typically small, e.g., 0.01)
 * @return x if x > 0, alpha * x otherwise
 */
static inline float lrelu(float x, float alpha)
{
    // Avoid branching for better performance
    float mask = x > 0.0f ? 1.0f : alpha;
    return x * mask;
}

/**
 * @brief SIMD-optimized Leaky ReLU for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 * @param alpha Slope for negative values
 */
static inline void lrelu_vector(float *input, float *output, int size, float alpha)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    __m128 zero = _mm_setzero_ps();
    __m128 alpha_vec = _mm_set1_ps(alpha);
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        
        // Create mask where x > 0
        __m128 mask = _mm_cmpgt_ps(x, zero);
        
        // Select alpha*x or x based on mask
        __m128 alpha_x = _mm_mul_ps(alpha_vec, x);
        __m128 result = _mm_or_ps(
            _mm_and_ps(mask, x),                // x where x > 0
            _mm_andnot_ps(mask, alpha_x)        // alpha*x where x <= 0
        );
        
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = lrelu(input[i], alpha);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = lrelu(input[i], alpha);
        output[i+1] = lrelu(input[i+1], alpha);
        output[i+2] = lrelu(input[i+2], alpha);
        output[i+3] = lrelu(input[i+3], alpha);
    }
    
    for (; i < size; i++) {
        output[i] = lrelu(input[i], alpha);
    }
#endif
}

/**
 * @brief Parametric ReLU activation function
 * @param x Input value
 * @param alpha Learnable parameter for negative values
 * @return x if x > 0, alpha * x otherwise
 */
static inline float prelu(float x, float alpha)
{
    // Implementation is identical to lrelu
    float mask = x > 0.0f ? 1.0f : alpha;
    return x * mask;
}

/**
 * @brief SIMD-optimized Parametric ReLU for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 * @param alpha Array of alpha values (one per neuron)
 */
static inline void prelu_vector(float *input, float *output, int size, float *alpha)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    __m128 zero = _mm_setzero_ps();
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 alpha_vec = _mm_loadu_ps(&alpha[i]);
        
        // Create mask where x > 0
        __m128 mask = _mm_cmpgt_ps(x, zero);
        
        // Select alpha*x or x based on mask
        __m128 alpha_x = _mm_mul_ps(alpha_vec, x);
        __m128 result = _mm_or_ps(
            _mm_and_ps(mask, x),                // x where x > 0
            _mm_andnot_ps(mask, alpha_x)        // alpha*x where x <= 0
        );
        
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = prelu(input[i], alpha[i]);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = prelu(input[i], alpha[i]);
        output[i+1] = prelu(input[i+1], alpha[i+1]);
        output[i+2] = prelu(input[i+2], alpha[i+2]);
        output[i+3] = prelu(input[i+3], alpha[i+3]);
    }
    
    for (; i < size; i++) {
        output[i] = prelu(input[i], alpha[i]);
    }
#endif
}

/**
 * @brief Exponential Linear Unit activation function
 * @param x Input value
 * @param alpha Scale for the negative part
 * @return x if x ≥ 0, alpha * (exp(x) - 1) otherwise
 */
static inline float elu(float x, float alpha)
{
    // Avoid branching for better performance
    if (x >= 0.0f) {
        return x;
    } else {
        return alpha * (fast_exp(x) - 1.0f);
    }
}

/**
 * @brief SIMD-optimized ELU for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 * @param alpha Scale parameter for negative values
 */
static inline void elu_vector(float *input, float *output, int size, float alpha)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    __m128 zero = _mm_setzero_ps();
    __m128 alpha_vec = _mm_set1_ps(alpha);
    __m128 one = _mm_set1_ps(1.0f);
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        
        // Create mask where x >= 0
        __m128 mask = _mm_cmpge_ps(x, zero);
        
        // Calculate exp(x) - 1 for each value
        float temp[4];
        _mm_storeu_ps(temp, x);
        
        __m128 exp_minus_one = _mm_set_ps(
            fast_exp(temp[3]) - 1.0f,
            fast_exp(temp[2]) - 1.0f,
            fast_exp(temp[1]) - 1.0f,
            fast_exp(temp[0]) - 1.0f
        );
        
        __m128 neg_result = _mm_mul_ps(alpha_vec, exp_minus_one);
        
        // Select x or alpha*(exp(x)-1) based on mask
        __m128 result = _mm_or_ps(
            _mm_and_ps(mask, x),                // x where x >= 0
            _mm_andnot_ps(mask, neg_result)     // alpha*(exp(x)-1) where x < 0
        );
        
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = elu(input[i], alpha);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = elu(input[i], alpha);
        output[i+1] = elu(input[i+1], alpha);
        output[i+2] = elu(input[i+2], alpha);
        output[i+3] = elu(input[i+3], alpha);
    }
    
    for (; i < size; i++) {
        output[i] = elu(input[i], alpha);
    }
#endif
}

/**
 * @brief Softmax activation function for entire layer with enhanced numerical stability
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 */
static inline void softmax(float *input, float *output, int size)
{
    // Find maximum for numerical stability
    float max_val = input[0];
    
    // Find max with loop unrolling for better instruction pipelining
    int i = 1;
    int unroll_size = size & ~3; // Round down to multiple of 4
    
    for (; i < unroll_size; i += 4) {
        max_val = input[i] > max_val ? input[i] : max_val;
        max_val = input[i+1] > max_val ? input[i+1] : max_val;
        max_val = input[i+2] > max_val ? input[i+2] : max_val;
        max_val = input[i+3] > max_val ? input[i+3] : max_val;
    }
    
    for (; i < size; i++) {
        max_val = input[i] > max_val ? input[i] : max_val;
    }

    // Calculate sum of exponentials with shift for stability
    float sum = 0.0f;
    
    #ifdef USE_SIMD
    __m128 sum_vec = _mm_setzero_ps();
    __m128 max_vec = _mm_set1_ps(max_val);
    int simd_size = size & ~3;
    
    // Calculate exponentials and sum in parallel
    for (i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 shifted = _mm_sub_ps(x, max_vec);
        
        // Calculate exp() for each component
        float temp[4];
        _mm_storeu_ps(temp, shifted);
        
        __m128 exp_result = _mm_set_ps(
            fast_exp(temp[3]),
            fast_exp(temp[2]),
            fast_exp(temp[1]),
            fast_exp(temp[0])
        );
        
        // Store intermediate exponential results
        _mm_storeu_ps(&output[i], exp_result);
        
        // Accumulate sum
        sum_vec = _mm_add_ps(sum_vec, exp_result);
    }
    
    // Extract sum from vector
    float sum_array[4];
    _mm_storeu_ps(sum_array, sum_vec);
    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (i = simd_size; i < size; i++) {
        output[i] = fast_exp(input[i] - max_val);
        sum += output[i];
    }
    #else
    // Scalar version with loop unrolling
    i = 0;
    for (; i < unroll_size; i += 4) {
        output[i] = fast_exp(input[i] - max_val);
        output[i+1] = fast_exp(input[i+1] - max_val);
        output[i+2] = fast_exp(input[i+2] - max_val);
        output[i+3] = fast_exp(input[i+3] - max_val);
        
        sum += output[i] + output[i+1] + output[i+2] + output[i+3];
    }
    
    for (; i < size; i++) {
        output[i] = fast_exp(input[i] - max_val);
        sum += output[i];
    }
    #endif

    // Normalize - using multiplication by reciprocal (faster than division)
    float inv_sum = 1.0f / sum;
    
    #ifdef USE_SIMD
    __m128 inv_sum_vec = _mm_set1_ps(inv_sum);
    
    for (i = 0; i < simd_size; i += 4) {
        __m128 exp_vals = _mm_loadu_ps(&output[i]);
        __m128 normalized = _mm_mul_ps(exp_vals, inv_sum_vec);
        _mm_storeu_ps(&output[i], normalized);
    }
    
    for (; i < size; i++) {
        output[i] *= inv_sum;
    }
    #else
    // Normalization with loop unrolling
    i = 0;
    for (; i < unroll_size; i += 4) {
        output[i] *= inv_sum;
        output[i+1] *= inv_sum;
        output[i+2] *= inv_sum;
        output[i+3] *= inv_sum;
    }
    
    for (; i < size; i++) {
        output[i] *= inv_sum;
    }
    #endif
}

/**
 * @brief Single-input softmax for network structure
 * @param x Input value
 * @return Exponential of input (partial softmax)
 */
static inline float softmax_single(float x)
{
    return fast_exp(x);
}

/**
 * @brief Derivative of softmax function
 * @param x Input value
 * @return Derivative value for backpropagation
 */
static inline float softmax_derivative(float x)
{
    float s = softmax_single(x);
    return s * (1.0f - s);
}

/**
 * @brief Gaussian Error Linear Unit (GELU) activation with fast approximation
 * @param x Input value
 * @return GELU activation value
 */
static inline float gelu(float x)
{
    // Fast GELU approximation: x * sigmoid(1.702 * x)
    return x * sigmoid(1.702f * x);
}

/**
 * @brief SIMD-optimized GELU for array processing
 * @param input Array of input values
 * @param output Array to store results
 * @param size Length of input/output arrays
 */
static inline void gelu_vector(float *input, float *output, int size)
{
#ifdef USE_SIMD
    int simd_size = size & ~3;
    __m128 coef = _mm_set1_ps(1.702f);
    
    for (int i = 0; i < simd_size; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 scaled_x = _mm_mul_ps(coef, x);
        
        // Calculate sigmoid for each component
        float temp[4];
        _mm_storeu_ps(temp, scaled_x);
        
        __m128 sig_result = _mm_set_ps(
            sigmoid(temp[3]),
            sigmoid(temp[2]),
            sigmoid(temp[1]),
            sigmoid(temp[0])
        );
        
        // Multiply by x
        __m128 result = _mm_mul_ps(x, sig_result);
        _mm_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = simd_size; i < size; i++) {
        output[i] = gelu(input[i]);
    }
#else
    // Scalar fallback with loop unrolling
    int i = 0;
    int unroll_size = size & ~3;
    
    for (; i < unroll_size; i += 4) {
        output[i] = gelu(input[i]);
        output[i+1] = gelu(input[i+1]);
        output[i+2] = gelu(input[i+2]);
        output[i+3] = gelu(input[i+3]);
    }
    
    for (; i < size; i++) {
        output[i] = gelu(input[i]);
    }
#endif
}

#endif // ACTIVATION_FUNCTIONS_H