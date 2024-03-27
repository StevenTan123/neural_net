#include "layer.h"
#include "loss.h"
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

Loss::Loss(int _out_size) : out_size(_out_size) {
    grads = new double[out_size];
    for (int i = 0; i < out_size; i++) {
        grads[i] = 0;
    }
}

double MeanSquaredLoss::calc_loss(double *output, double *expected) {
    double error = 0;
    for (int i = 0; i < out_size; i++) {
        error += (output[i] - expected[i]) * (output[i] - expected[i]);
        grads[i] = (2 * output[i] - 2 * expected[i]) / out_size;
    }
    error /= out_size;
    return error;
}

