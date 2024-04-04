#include "MNIST_loader.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using std::ifstream;
using std::string;
using std::vector;
using std::stringstream;

MNIST_loader::MNIST_loader(string filename) {
    ifstream file(filename);
    string line;
    
    int row = 0;
    while (std::getline(file, line)) {
        if (row == 0) {
            row++;
            continue;
        }

        int col = 0;
        double *img = new double[IMG_LEN];
        double *label = new double[N_CLASSES];
        for (int i = 0; i < N_CLASSES; i++) {
            label[i] = 0;
        }
        
        stringstream ss(line);
        string token;
        while (std::getline(ss, token, ',')) {
            int val = std::stoi(token);
            if (col == 0) {
                label[val] = 1;
            } else {
                img[col - 1] = val;
            }
            col++;
        }
        data.push_back(img);
        labels.push_back(label);
        row++;
    }
}

MNIST_loader::~MNIST_loader() {
    for (double *input : data) {
        delete[] input;
    }
    for (double *label : labels) {
        delete[] label;
    }
}
