#include "NeuralNetwork.hpp"

__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k){
        float sum = 0.0f;
        for(int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

__global__ void matmul_a_bt_kernel(float* A, float* B, float* C, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < k){
        float sum = 0.0f;
        for(int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[col * n + i];
        C[row * k + col] = sum;
    }
}

__global__ void matmul_at_b_kernel(float* A, float* B, float* C, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < k){
        float sum = 0.0f;
        for(int i = 0; i < m; ++i)
            sum += A[i * n + row] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

__global__ void relu_kernel(float* x, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        x[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if(b < batch_size && i < size)
        x[idx] += bias[i];
}

__global__ void softmax_kernel(float* x, int batch_size, int size){
    int b = blockIdx.x;
    if(b < batch_size){
        float max_val = x[b * size];
        for(int i = 1; i < size; ++i)
            max_val = fmaxf(max_val, x[b * size + i]);

        float sum = 0.0f;
        for(int i = 0; i < size; ++i){
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for(int i = 0; i < size; ++i)
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
    }
}

__global__ void clip_gradients_kernel(float* gradients, int size, float max_norm){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        float grad = gradients[idx];
        if(grad > max_norm) 
            gradients[idx] = max_norm;
        else if(grad < -max_norm)
            gradients[idx] = -max_norm;
    }
}

__global__ void zero_grad_kernel(float* grad, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        grad[idx] = 0.0f;
}

__global__ void compute_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size){
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b < batch_size){
        for(int i = 0; i < OUTPUT_DIM; ++i)
            grad_output[b * OUTPUT_DIM + i] = output[b * OUTPUT_DIM + i];
        grad_output[b * OUTPUT_DIM + labels[b]] -= 1.0f;
    }
}

__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size){
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < curr_size && j < prev_size){
        float grad_w_sum = 0.0f;
        for(int b = 0; b < batch_size; ++b)
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

        if(j == 0){
            float grad_b_sum = 0.0f;
            for(int b = 0; b < batch_size; ++b)
                grad_b_sum += grad_layer[b * curr_size + i];
            atomicAdd(&grad_bias[i], grad_b_sum);
        }
    }
}

__global__ void drelu_kernel(float* x, float* d_ReLU_out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        d_ReLU_out[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;
}

__global__ void multiply_gradients_kernel(float* grad1, float* grad2, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        grad1[idx] *= grad2[idx];
}

__global__ void update_weights_kernel(float* weights, float* grad_weights, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
}