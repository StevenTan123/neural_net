#ifndef LAYER_H
#define LAYER_H

// Stores a layer of the neural network. Abstract class.
class Layer {
public:
    // Input and output layer sizes.
    int in_size, out_size;
    
    // in_grads[i] stores gradient of the i-th input node.
    double *in_grads;
    
    // Values of input and output nodes.
    double *input;
    double *output;
    
    Layer(int _in_size, int _out_size); 
    virtual ~Layer();

    // Forward propagation. Takes input layer values, produces output layer values.
    virtual void forward(double *input) = 0;

    // Backpropagation. Updates in_grads given out_grad.
    virtual void backprop(double *out_grads) = 0;

    // Reset all gradients to 0.
    virtual void reset_grad();

    // Steps weights towards negative gradient.
    virtual void step(double learning_rate) = 0;
};

// Fully connected linear layer. Inherits from Layer. Additionally stores edge weights leading 
// into current layer (and their gradients), and also biases (and their gradients).
class LinearLayer : public Layer {
public:
    // weights[i][j] = weight of the edge from the j-th node of input layer to i-th node of 
    // output layer. w_grads[i][j] stores the gradients of those edges. 
    double **weights;
    double **w_grads;

    // biases[i] = bias of the i-th node of output layer. bias_grads[i] stores gradient of
    // that bias.
    double *biases;
    double *bias_grads;

    LinearLayer(int _in_size, int _out_size); 
    ~LinearLayer();

    void forward(double *input) override;
    void backprop(double *out_grads) override;
    void reset_grad() override;
    void step(double learning_rate) override;
};

// Activation layer using tanh function.
class TanhLayer : public Layer {
public:
    TanhLayer(int size) : Layer(size, size) {} 
    ~TanhLayer() {}

    void forward(double *input) override;
    void backprop(double *out_grads) override;
    // Step does nothing for activation layers, do useless operation to prevent compiler warning.
    void step(double learning_rate) override { learning_rate++; };
};

#endif