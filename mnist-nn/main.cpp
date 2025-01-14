#include "NeuralNetwork.hpp"

void load_input_data(const char* file_name, float* data, int size) {
    FILE* file = fopen(file_name, "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        std::cerr << "Error reading data: expected " << size << " elements, read " << read_size << std::endl;
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

void load_labels(const char* file_name, int* labels, int size) {
    FILE* file = fopen(file_name, "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        std::cerr << "Error reading labels: expected " << size << " elements, read " << read_size << std::endl;
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;

    float* X_train = new float[TRAIN_SIZE * INPUT_DIM];
    int* y_train = new int[TRAIN_SIZE];
    float* X_test = new float[TEST_SIZE * INPUT_DIM];
    int* y_test = new int[TEST_SIZE];

    load_input_data("mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_DIM);
    load_labels("mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_input_data("mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_DIM);
    load_labels("mnist_data/y_test.bin", y_test, TEST_SIZE);

    nn.train(X_train, y_train);
    nn.test(X_test, y_test);

    delete[] X_train;
    delete[] y_train;
    delete[] X_test;
    delete[] y_test;

    std::cout << "Success" << std::endl;
    return 0;
}
