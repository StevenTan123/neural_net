#include "neural_net.h"
#include <iostream>
#include <vector>
#include <random>

using std::cin;
using std::cout;
using std::endl;
using std::vector;

NeuralNet::~NeuralNet() {
    for (Layer *layer : layers) {
        delete layer;
    }
    delete loss;
}

double *NeuralNet::forward(double *input) {
    for (Layer *layer : layers) {
        layer->forward(input);
        input = layer->output;
    }
    return input;
}

void NeuralNet::backprop(double *out_grads) {
    for (int i = layers.size() - 1; i >= 0; i--) {
        layers[i]->backprop(out_grads);
        out_grads = layers[i]->in_grads;       
    }
}

void NeuralNet::step(double learning_rate) {
    for (Layer *layer : layers) {
        layer->step(learning_rate);
    }
}

void NeuralNet::reset_grad() {
    for (Layer *layer : layers) {
        layer->reset_grad();
    }
}

void NeuralNet::fit(vector<double *> input_train, vector<double *> output_train, int epochs, int batch_size, double learning_rate) {
    int train_size = input_train.size();
    int batches = (train_size / batch_size) + (train_size % batch_size == 0 ? 0 : 1);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double avg_error = 0;
        
        for (int batch = 0; batch < batches; batch++) {
            int from = batch * batch_size;
            int to = std::min(from + batch_size, train_size);
            double batch_error = 0;
            
            for (int i = from; i < to; i++) {
                double *cur_output = forward(input_train[i]);
                double error = loss->calc_loss(cur_output, output_train[i]);
                avg_error += error;
                batch_error += error;
                
                backprop(loss->grads);
            }
            batch_error /= to - from;
            
            step(learning_rate);
            reset_grad();

            if (batch > 0 && batch % 500 == 0) cout << "Batch #" << batch << ", Error: " << batch_error << endl; 
        }

        avg_error /= train_size;
        cout << "Epoch #" << epoch << ", Error: " << avg_error << endl; 
    }
}

void NeuralNet::print_nn() {
    cout << "============================================" << endl;
    int cnt = 0;
    for (Layer *layer : layers) {
        LinearLayer *cast = dynamic_cast<LinearLayer*>(layer);
        if (cast != nullptr) {
            cout << "Layer #" << cnt << endl;
            for (int i = 0; i < cast->out_size; i++) {
                for (int j = 0; j < cast->in_size; j++) {
                    cout << cast->w_grads[i][j] << " ";
                }
                cout << endl;
            }
            cnt++;
        }
    }
}
