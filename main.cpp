#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include "neural_net.h"
#include "loss.h"

using std::vector;
using std::cin;
using std::cout;
using std::endl;

int main() {

    // Creating neural network and adding layers.
    NeuralNet *nn = new NeuralNet();
    nn->add_layer(new LinearLayer(1, 16));
    nn->add_layer(new TanhLayer(16));
    nn->add_layer(new LinearLayer(16, 4));
    nn->add_layer(new TanhLayer(4));
    nn->add_layer(new LinearLayer(4, 1));
    nn->add_layer(new TanhLayer(1));
    nn->add_loss(new MeanSquaredLoss(1));

    // Setting up input and output training data. Here we are trying to fit the sin function.
    int N = 64;
    vector<double*> inputs(N);
    vector<double*> outputs(N);
    for (int i = 0; i < N; i++) {
        inputs[i] = new double[1];
        inputs[i][0] = 2 * M_PI / N * i;
        outputs[i] = new double[1];
        outputs[i][0] = sin(inputs[i][0]);
    }
    
    /*
    NeuralNet *nn = new NeuralNet();
    nn->add_layer(new LinearLayer(2, 3));
    nn->add_layer(new TanhLayer(3));
    nn->add_layer(new LinearLayer(3, 1));
    nn->add_layer(new TanhLayer(1));
    nn->add_loss(new MeanSquaredLoss(1));
    
    int N = 4;
    vector<double*> inputs(4);
    inputs[0] = new double[2] {0, 0};
    inputs[1] = new double[2] {0, 1};
    inputs[2] = new double[2] {1, 0};
    inputs[3] = new double[2] {1, 1};
    vector<double*> outputs(4);
    outputs[0] = new double[1] {0};
    outputs[1] = new double[1] {1};
    outputs[2] = new double[1] {1};
    outputs[3] = new double[1] {0};
    */

    // Fit the neural network.
    nn->fit(inputs, outputs, 1000, 0.1);

    for (int i = 0; i < N; i++) {
        double *output = nn->forward(inputs[i]);
        cout << "Input: " << inputs[i][0] << ", Output: " << output[0] << endl;
    }

    // Free any allocated memory.
    for (int i = 0; i < N; i++) {
        delete[] inputs[i];
        delete[] outputs[i];
    }
    delete nn;
}