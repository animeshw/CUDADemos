#include <iostream>
#include "NeuralNetwork.hpp"

#define CUDA_CHECK(call) check_cuda_error(call, #call)

NeuralNetwork::NeuralNetwork() {
    CUDA_CHECK(cudaMalloc(&d_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_hidden1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_hidden2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_output, OUTPUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_grad_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_hidden1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_hidden2, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_output, OUTPUT_DIM * sizeof(float)));

    float* h_weights_input_to_hidden1 = new float[HIDDEN_SIZE * INPUT_DIM];
    float* h_weights_hidden1_to_hidden2 = new float[HIDDEN_SIZE * HIDDEN_SIZE];
    float* h_weights_hidden2_to_output = new float[OUTPUT_DIM * HIDDEN_SIZE];
    float* h_bias_hidden1 = new float[HIDDEN_SIZE];
    float* h_bias_hidden2 = new float[HIDDEN_SIZE];
    float* h_bias_output = new float[OUTPUT_DIM];

    initialize_weights(h_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM);
    initialize_weights(h_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE);
    initialize_weights(h_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE);
    initialize_bias(h_bias_hidden1, HIDDEN_SIZE);
    initialize_bias(h_bias_hidden2, HIDDEN_SIZE);
    initialize_bias(h_bias_output, OUTPUT_DIM);

    CUDA_CHECK(cudaMemcpy(d_weights_input_to_hidden1, h_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights_hidden1_to_hidden2, h_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights_hidden2_to_output, h_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_hidden1, h_bias_hidden1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_hidden2, h_bias_hidden2, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_output, h_bias_output, OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_weights_input_to_hidden1;
    delete[] h_weights_hidden1_to_hidden2;
    delete[] h_weights_hidden2_to_output;
    delete[] h_bias_hidden1;
    delete[] h_bias_hidden2;
    delete[] h_bias_output;
}

NeuralNetwork::~NeuralNetwork() {
    CUDA_CHECK(cudaFree(d_weights_input_to_hidden1));
    CUDA_CHECK(cudaFree(d_weights_hidden1_to_hidden2));
    CUDA_CHECK(cudaFree(d_weights_hidden2_to_output));
    CUDA_CHECK(cudaFree(d_bias_hidden1));
    CUDA_CHECK(cudaFree(d_bias_hidden2));
    CUDA_CHECK(cudaFree(d_bias_output));
    CUDA_CHECK(cudaFree(d_grad_weights_input_to_hidden1));
    CUDA_CHECK(cudaFree(d_grad_weights_hidden1_to_hidden2));
    CUDA_CHECK(cudaFree(d_grad_weights_hidden2_to_output));
    CUDA_CHECK(cudaFree(d_grad_bias_hidden1));
    CUDA_CHECK(cudaFree(d_grad_bias_hidden2));
    CUDA_CHECK(cudaFree(d_grad_bias_output));
}

void NeuralNetwork::initialize_weights(float* weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; ++i)
        weights[i] = (static_cast<float>(rand()) / RAND_MAX) * scale - (scale / 2.0f);
}

void NeuralNetwork::initialize_bias(float* bias, int size) {
    for (int i = 0; i < size; ++i)
        bias[i] = 0.0f;
}

void NeuralNetwork::check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void NeuralNetwork::forward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int batch_size){
    dim3 block_size(32, 32);
    dim3 grid_size;

    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_input, d_weights_input_to_hidden1, d_hidden1, batch_size, INPUT_DIM, HIDDEN_SIZE);
    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden1, d_bias_hidden1, batch_size, HIDDEN_SIZE);
    relu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden1, batch_size * HIDDEN_SIZE);

    matmul_a_b_kernel<<<grid_size, block_size>>>(d_hidden1, d_weights_hidden1_to_hidden2, d_hidden2, batch_size, HIDDEN_SIZE, HIDDEN_SIZE);
    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden2, d_bias_hidden2, batch_size, HIDDEN_SIZE);
    relu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden2, batch_size * HIDDEN_SIZE);

    grid_size.x = (OUTPUT_DIM + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_hidden2, d_weights_hidden2_to_output, d_output, batch_size, HIDDEN_SIZE, OUTPUT_DIM);
    bias_add_kernel<<<(batch_size * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_bias_output, batch_size, OUTPUT_DIM);
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_DIM);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::backward_propagation(float* d_input, float* d_hidden1, float* d_hidden2, float* d_output, int* d_labels, int batch_size){
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_DIM + 256 - 1) / 256, 256>>>(d_grad_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM);
    zero_grad_kernel<<<(HIDDEN_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_DIM * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_bias_hidden1, HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_bias_hidden2, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_DIM + 256 - 1) / 256, 256>>>(d_grad_bias_output, OUTPUT_DIM);

    float* d_grad_output = NULL;
    float* d_dX2 = NULL;
    float* d_dX3 = NULL;
    float* d_grad_hidden1 = NULL;
    float* d_grad_hidden2 = NULL;

    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dX3, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden1, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden2, batch_size * HIDDEN_SIZE * sizeof(float)));

    dim3 block_size(32, 32);
    dim3 grid_size;
    
    compute_gradients_kernel<<<(batch_size + 256 - 1) / 256, 256>>>(d_grad_output, d_output, d_labels, batch_size);

    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (OUTPUT_DIM + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_hidden2, d_grad_output, d_grad_weights_hidden2_to_output, batch_size, HIDDEN_SIZE, OUTPUT_DIM);
    update_gradients_kernel<<<grid_size, block_size>>>(d_grad_weights_hidden2_to_output, d_grad_bias_output, d_grad_output, d_hidden2, batch_size, HIDDEN_SIZE, OUTPUT_DIM);

    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_bt_kernel<<<grid_size, block_size>>>(d_grad_output, d_weights_hidden2_to_output, d_dX3, batch_size, OUTPUT_DIM, HIDDEN_SIZE);
    drelu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden2, d_grad_hidden2, batch_size * HIDDEN_SIZE);
    multiply_gradients_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_dX3, d_grad_hidden2, batch_size * HIDDEN_SIZE);

    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_SIZE + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_hidden1, d_dX3, d_grad_weights_hidden1_to_hidden2, batch_size, HIDDEN_SIZE, HIDDEN_SIZE);
    update_gradients_kernel<<<grid_size, block_size>>>(d_grad_weights_hidden1_to_hidden2, d_grad_bias_hidden2, d_dX3, d_hidden1, batch_size, HIDDEN_SIZE, HIDDEN_SIZE);

    matmul_a_bt_kernel<<<grid_size, block_size>>>(d_dX3, d_weights_hidden1_to_hidden2, d_dX2, batch_size, HIDDEN_SIZE, HIDDEN_SIZE);
    drelu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden1, d_grad_hidden1, batch_size * HIDDEN_SIZE);
    multiply_gradients_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_dX2, d_grad_hidden1, batch_size * HIDDEN_SIZE);

    grid_size.x = (INPUT_DIM + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_SIZE + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_input, d_dX2, d_grad_weights_input_to_hidden1, batch_size, INPUT_DIM, HIDDEN_SIZE);
    update_gradients_kernel<<<grid_size, block_size>>>(d_grad_weights_input_to_hidden1, d_grad_bias_hidden1, d_dX2, d_input, batch_size, INPUT_DIM, HIDDEN_SIZE);

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_dX2));
    CUDA_CHECK(cudaFree(d_dX3));
    CUDA_CHECK(cudaFree(d_grad_hidden1));
    CUDA_CHECK(cudaFree(d_grad_hidden2));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::update_weights(){
    int block_size = 256;
    int grid_size;

    grid_size = (HIDDEN_SIZE * INPUT_DIM + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_weights_input_to_hidden1, d_grad_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM);
    CUDA_CHECK(cudaGetLastError());

    grid_size = (HIDDEN_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_weights_hidden1_to_hidden2, d_grad_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    grid_size = (HIDDEN_SIZE * OUTPUT_DIM + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_weights_hidden2_to_output, d_grad_weights_hidden2_to_output, HIDDEN_SIZE * OUTPUT_DIM);
    CUDA_CHECK(cudaGetLastError());

    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_bias_hidden1, d_grad_bias_hidden1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_bias_hidden2, d_grad_bias_hidden2, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    grid_size = (OUTPUT_DIM + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(d_bias_output, d_grad_bias_output, OUTPUT_DIM);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

float NeuralNetwork::cross_entropy_loss(float* output, int* labels, int batch_size){
    float total_loss = 0.0f;
    for(int b = 0; b < batch_size; b++)
        total_loss -= logf(fmaxf(output[b * OUTPUT_DIM + labels[b]], 1e-7f));
    return total_loss / batch_size;
}

void NeuralNetwork::train(float* training_data, int* training_labels){
    float *d_X_train = NULL;
    float *d_hidden1 = NULL;
    float *d_hidden2 = NULL;
    float *d_output = NULL;
    int *d_y_train =NULL;

    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_X_train, training_data, TRAIN_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, training_labels, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for(int epoch = 0; epoch < EPOCHS; ++epoch){
        float total_loss = 0.0f;
        int correct = 0;

        zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_DIM + 256 - 1) / 256, 256>>>(d_grad_weights_input_to_hidden1, HIDDEN_SIZE * INPUT_DIM);
        zero_grad_kernel<<<(HIDDEN_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_weights_hidden1_to_hidden2, HIDDEN_SIZE * HIDDEN_SIZE);
        zero_grad_kernel<<<(OUTPUT_DIM * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_weights_hidden2_to_output, OUTPUT_DIM * HIDDEN_SIZE);
        zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_bias_hidden1, HIDDEN_SIZE);
        zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(d_grad_bias_hidden2, HIDDEN_SIZE);
        zero_grad_kernel<<<(OUTPUT_DIM + 256 - 1) / 256, 256>>>(d_grad_bias_output, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        for(int batch = 0; batch < num_batches; ++batch){
            int start_idx = batch * BATCH_SIZE;

            forward_propagation(&d_X_train[start_idx * INPUT_DIM], d_hidden1, d_hidden2, d_output, BATCH_SIZE);

            float* h_output = new float[BATCH_SIZE * OUTPUT_DIM];
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));

            float loss = cross_entropy_loss(h_output, &training_labels[start_idx], BATCH_SIZE);
            total_loss += loss;

            for(int i = 0; i < BATCH_SIZE; ++i){
                int predicted = 0;
                for(int j = 1; j < OUTPUT_DIM; ++j){
                    if(h_output[i * OUTPUT_DIM + j] > h_output[i * OUTPUT_DIM + predicted])
                        predicted = j;
                }
                if(predicted == training_labels[start_idx + i])
                    ++correct;
            }

            delete[] h_output;

            backward_propagation(&d_X_train[start_idx * INPUT_DIM], d_hidden1, d_hidden2, d_output, &d_y_train[start_idx], BATCH_SIZE);

            update_weights();

            if((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)){
                std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << ", Iter " << batch + 1 << "/" << num_batches;
                std::cout << ", Loss " << total_loss / (batch + 1) << ", Accuracy " << 100.0f * correct / ((batch + 1) * BATCH_SIZE) << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_hidden1));
    CUDA_CHECK(cudaFree(d_hidden2));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_train));
}

void NeuralNetwork::test(float* test_data, int* test_labels){
    float* d_X_test = NULL;
    float* d_hidden1 = NULL;
    float* d_hidden2 = NULL;
    float* d_output = NULL;

    CUDA_CHECK(cudaMalloc(&d_X_test, NUM_TEST_SAMPLES * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden1, NUM_TEST_SAMPLES * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden2, NUM_TEST_SAMPLES * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, NUM_TEST_SAMPLES * OUTPUT_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X_test, test_data, NUM_TEST_SAMPLES * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));

    forward_propagation(d_X_test, d_hidden1, d_hidden2, d_output, NUM_TEST_SAMPLES);

    float* h_output = new float[NUM_TEST_SAMPLES * OUTPUT_DIM];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, NUM_TEST_SAMPLES * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    int correct = 0;
    std::cout << "Test Results for first " << NUM_TEST_SAMPLES << " samples:" << std::endl;

    for(int i = 0; i < NUM_TEST_SAMPLES; ++i){
        std::cout << "Sample " << i + 1 << " : " << std::endl;
        for(int row = 0; row < 28; ++row){
            for(int col = 0; col < 28; ++col) {
                if(test_data[i * INPUT_DIM + row * 28 + col] > 0.0f)
                    std::cout << "X";
                else
                    std::cout << " ";
            }
            std::cout << std::endl;
        }

        int predicted = 0;
        for(int j = 1; j < OUTPUT_DIM; j++){
            if(h_output[i * OUTPUT_DIM + j] > h_output[i * OUTPUT_DIM + predicted])
                predicted = j;
        }

        if(predicted == test_labels[i])
            correct++;

        std::cout << "True Label: "<< test_labels[i] << ", Predicted: " << predicted << " " << ((predicted == test_labels[i]) ? "Right" : "Wrong") << std::endl;
    }

    std::cout << "Test Accuracy (first " << NUM_TEST_SAMPLES << " samples): " << (100.0f * correct / NUM_TEST_SAMPLES)<< std::endl;

    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_hidden1));
    CUDA_CHECK(cudaFree(d_hidden2));
    CUDA_CHECK(cudaFree(d_output));
    
    delete[] h_output;
}
