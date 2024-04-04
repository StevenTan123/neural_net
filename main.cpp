#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include "neural_net.h"
#include "loss.h"
#include "MNIST_loader.h"

using std::vector;
using std::cin;
using std::cout;
using std::endl;

NeuralNet *train_digit_classifier() {
    MNIST_loader train_data("data/mnist_train.csv");
    
    // Creating neural network and adding layers.
    NeuralNet *nn = new NeuralNet();
    nn->add_layer(new LinearLayer(784, 100));
    nn->add_layer(new TanhLayer(100));
    nn->add_layer(new LinearLayer(100, 50));
    nn->add_layer(new TanhLayer(50));
    nn->add_layer(new LinearLayer(50, 10));
    nn->add_layer(new TanhLayer(10));
    nn->add_loss(new MeanSquaredLoss(10));

    nn->fit(train_data.data, train_data.labels, 30, 32, 0.01);
    return nn;
}

void test_digit_classifier(NeuralNet *nn) {
    MNIST_loader test_data("data/mnist_train.csv");

    int correct = 0;
    int test_data_size = test_data.data.size();
    for (int i = 0; i < test_data_size; i++) {
        double *input = test_data.data[i];
        double *output = nn->forward(input);
        double *label = test_data.labels[i];

        int pred = 0;
        for (int j = 0; j < test_data.N_CLASSES; j++) {
            if (output[j] > output[pred]) {
                pred = j;
            }
        }

        if (label[pred] > 0.5) {
            correct++;
        }
    }

    double accuracy = (double) correct / test_data_size;
    cout << "Model Accuracy: " << accuracy << endl;
}

NeuralNet *train_sin() {
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
    int N = 16;
    vector<double*> inputs(N);
    vector<double*> outputs(N);
    for (int i = 0; i < N; i++) {
        inputs[i] = new double[1];
        inputs[i][0] = 2 * M_PI / N * i;
        outputs[i] = new double[1];
        outputs[i][0] = sin(inputs[i][0]);
    }

    // Fit the neural network.
    nn->fit(inputs, outputs, 1000, 1, 0.1);

    // Display output of neural network for the given inputs.
    for (int i = 0; i < N; i++) {
        double *output = nn->forward(inputs[i]);
        cout << "Input: " << inputs[i][0] << ", Output: " << output[0] << endl;
    }

    // Free any allocated memory.
    for (int i = 0; i < N; i++) {
        delete[] inputs[i];
        delete[] outputs[i];
    }
    return nn;
}

NeuralNet *train_xor() {
    // Creating neural network and adding layers.
    NeuralNet *nn = new NeuralNet();
    nn->add_layer(new LinearLayer(2, 8));
    nn->add_layer(new TanhLayer(8));
    nn->add_layer(new LinearLayer(8, 1));
    nn->add_layer(new TanhLayer(1));
    nn->add_loss(new MeanSquaredLoss(1));

    // Inputs and expected outputs for XOR. Used to train model.
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

    // Fit the neural network.
    nn->fit(inputs, outputs, 1000, 4, 0.1);

    // Display output of neural network for the given inputs. 
    for (int i = 0; i < 4; i++) {
        double *output = nn->forward(inputs[i]);
        cout << "Input: " << inputs[i][0] << inputs[i][1] << ", Output: " << output[0] << endl;
    }

    // Free any allocated memory.
    for (int i = 0; i < 4; i++) {
        delete[] inputs[i];
        delete[] outputs[i];
    }
    return nn;
}


int main() {
    NeuralNet *nn = train_digit_classifier();
    test_digit_classifier(nn);
    delete nn;
}
