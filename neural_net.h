#ifndef NEURALNET_H
#define NEURALNET_H
#include "layer.h"
#include "loss.h"
#include <vector>

class NeuralNet {
public:
    // Layers of the network.
    std::vector<Layer*> layers;
    // Loss function.
    Loss *loss;

    NeuralNet() {};
    ~NeuralNet();

    // Adds layer to the network.    
    void add_layer(Layer *layer) { layers.push_back(layer); }

    // Adds loss function to network.
    void add_loss(Loss *_loss) { loss = _loss; }

    // Given input values, feeds forward through network to calculate output.
    double *forward(double *input);

    // Backpropagation, calculates gradients in each layer, given gradients of
    // last output layer.
    void backprop(double *out_grads);

    // Steps weights towards negative gradient.
    void step(double learning_rate);

    // Resets all gradients.
    void reset_grad();

    // Fits neural network by minimizing loss through gradient descent.
    void fit(std::vector<double *> input_train, std::vector<double *> output_train, int epochs, double learning_rate);

    // Prints neural network weights.
    void print_nn();
};

#endif