Forward Propagation (Solid Arrows)                                  Backward Propagation (Dashed Arrows)

+-----------------+     +-----------------+     +-----------------+     +-----------------+     +-----------------+
|   Input Layer   | --> | Hidden Layer 1  | --> | Hidden Layer 2  | --> |  Output Layer   | --> | Loss Function   |
| (INPUT_DIM)     |     | (HIDDEN_SIZE)   |     | (HIDDEN_SIZE)   |     | (OUTPUT_DIM)    |     | (Cross-Entropy)|
+-----------------+     +-----------------+     +-----------------+     +-----------------+     +-----------------+





      ^                 ^                 ^                       ^                       ^
      |                 |                 |                       |                       |
      | Weights 1       | Weights 2       | Weights 3               |                       |
      | (INPUT_DIM x   | (HIDDEN_SIZE x   | (HIDDEN_SIZE x           |                       |
      |  HIDDEN_SIZE)  |  HIDDEN_SIZE)  |  OUTPUT_DIM)            |                       |
      |                 |                 |                       |                       |
      | Biases 1        | Biases 2        | Biases 3               |                       |
      | (HIDDEN_SIZE)   | (HIDDEN_SIZE)   | (OUTPUT_DIM)            |                       |
      v                 v                 v                       |                       |
      |                 |                 |                       |                       |
      | ReLU            | ReLU            | Softmax                 |                       |
      | Activation      | Activation      | Activation              |                       |
      +-----------------+     +-----------------+     +-----------------+                       |
                                                                                                |
                                                                                                |
                                                                                                <--------------------------------------------------
                                                                                                    | Gradient Calculation & Backpropagation
                                                                                                    | (Gradients of Loss w.r.t. Weights and Biases)
                                                                                                    |
                                                                                                    v
                                                                                                +-----------------+
                                                                                                | Weight Updates  |
                                                                                                | (using Gradients)|
                                                                                                +-----------------+
