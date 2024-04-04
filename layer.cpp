#include "layer.h"
#include <iostream>
#include <vector>
#include <random>

using std::cin;
using std::cout;
using std::endl;

// rng for initializing weights
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> rng(-0.3, 0.3);

Layer::Layer(int _in_size, int _out_size) : in_size(_in_size), out_size(_out_size) {
    in_grads = new double[in_size];
    for (int i = 0; i < in_size; i++) {
        in_grads[i] = 0;
    }
    input = new double[in_size];
    output = new double[out_size];
}

Layer::~Layer() {
    delete[] in_grads;
    delete[] input;
    delete[] output;
}

void Layer::reset_grad() {
    for (int i = 0; i < in_size; i++) {
        in_grads[i] = 0;
    }
}

LinearLayer::LinearLayer(int _in_size, int _out_size) : Layer(_in_size, _out_size) {
    weights = new double*[out_size];
    w_grads = new double*[out_size];
    for (int i = 0; i < out_size; i++) {
        weights[i] = new double[in_size];
        w_grads[i] = new double[in_size];
        for (int j = 0; j < in_size; j++) {
            weights[i][j] = rng(gen);
            w_grads[i][j] = 0;
        }
    }
    biases = new double[out_size];
    bias_grads = new double[out_size];
    for (int i = 0; i < out_size; i++) {
        biases[i] = rng(gen);
        bias_grads[i] = 0;
    }
}

LinearLayer::~LinearLayer() {
    for (int i = 0; i < out_size; i++) {
        delete[] weights[i];
        delete[] w_grads[i];
    }
    delete[] weights;
    delete[] w_grads;
    delete[] biases;
    delete[] bias_grads;
}

void LinearLayer::forward(double *input) {
    for (int i = 0; i < in_size; i++) {
        this->input[i] = input[i];
    }
    for (int i = 0; i < out_size; i++) {
        output[i] = biases[i];
        for (int j = 0; j < in_size; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

void LinearLayer::backprop(double *out_grads) {
    // The gradients for the input layer always need to be reset.
    for (int i = 0; i < in_size; i++) {
        in_grads[i] = 0;
    }
    for (int i = 0; i < out_size; i++) {
        bias_grads[i] += out_grads[i];
        for (int j = 0; j < in_size; j++) {
            w_grads[i][j] += input[j] * out_grads[i];
            in_grads[j] += weights[i][j] * out_grads[i];
        }
    }
}

void LinearLayer::reset_grad() {
    Layer::reset_grad();
    for (int i = 0; i < out_size; i++) {
        bias_grads[i] = 0;
        for (int j = 0; j < in_size; j++) {
            w_grads[i][j] = 0;
        }
    }
}

void LinearLayer::step(double learning_rate) {
    for (int i = 0; i < out_size; i++) {
        biases[i] -= bias_grads[i] * learning_rate;
        for (int j = 0; j < in_size; j++) {
            weights[i][j] -= w_grads[i][j] * learning_rate;
        }
    }
}

void TanhLayer::forward(double *input) {
    for (int i = 0; i < in_size; i++) {
        this->input[i] = input[i];
    }
    for (int i = 0; i < out_size; i++) {
        output[i] = tanh(input[i]);
    }
}

void TanhLayer::backprop(double *out_grads) {
    // The gradients for the input layer always need to be reset.
    for (int i = 0; i < in_size; i++) {
        in_grads[i] = 0;
    }
    for (int i = 0; i < out_size; i++) {
        in_grads[i] += (1 - output[i] * output[i]) * out_grads[i];
    }
}
