#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k);
__global__ void matmul_a_bt_kernel(float* A, float* B, float* C, int m, int n, int k);
__global__ void matmul_at_b_kernel(float* A, float* B, float* C, int m, int n, int k);
__global__ void relu_kernel(float* x, int size);
__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size);
__global__ void softmax_kernel(float* x, int batch_size, int size);
__global__ void clip_gradients_kernel(float* gradients, int size, float max_norm);
__global__ void zero_grad_kernel(float* grad, int size);
__global__ void compute_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size);
__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size);
__global__ void drelu_kernel(float* x, float* d_ReLU_out, int size);
__global__ void multiply_gradients_kernel(float* grad1, float* grad2, int size);
__global__ void update_weights_kernel(float* weights, float* grad_weights, int size);

const int INPUT_DIM = 784;
const int HIDDEN_SIZE = 1024;
const int OUTPUT_DIM = 10;
const int TRAIN_SIZE = 10000;
const int TEST_SIZE = 1000;
const int BATCH_SIZE = 4;
const int EPOCHS = 3;
const int NUM_TEST_SAMPLES = 20;

#define LEARNING_RATE 0.01

class NeuralNetwork{
    public:
        NeuralNetwork();
        ~NeuralNetwork();
        void train(float* training_data, int* training_labels);
        void test(float* test_data, int* test_labels);

    private:
        float *d_weights_input_to_hidden1, *d_weights_hidden1_to_hidden2, *d_weights_hidden2_to_output;
        float *d_bias_hidden1, *d_bias_hidden2, *d_bias_output;
        float *d_grad_weights_input_to_hidden1, *d_grad_weights_hidden1_to_hidden2, *d_grad_weights_hidden2_to_output;
        float *d_grad_bias_hidden1, *d_grad_bias_hidden2, *d_grad_bias_output;

        void initialize_weights(float* weights, int size);
        void initialize_bias(float* bias, int size);
        void check_cuda_error(cudaError_t err, const char* msg);
        void forward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int batch_size);
        void backward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int* d_labels, int batch_size);
        void update_weights();
        float cross_entropy_loss(float* output, int* labels, int batch_size);        
};

#endif // NEURAL_NETWORK_HPP
