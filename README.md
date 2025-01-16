## Neural Network Architecture

![NeuralNetwork](https://github.com/user-attachments/assets/92035ec2-f114-429f-96b6-9eb490bc38ec)

## Setup

1. Install cuda
2. Extract mnist_data.zip into mnist-nn folder
```bash
cd mnist-nn
nvcc -o app.exe main.cpp NeuralNetwork.cu kernels.cu
```

## Output
![Output](https://github.com/user-attachments/assets/312eed53-87ad-42a0-a72b-07957dad2276)
