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

MNIST_loader::MNIST_loader(string train_filename, string test_filename) {
    ifstream train_file(train_filename);
    ifstream test_file(test_filename);

    populate_data(train_file, train_data, train_labels);
    populate_data(test_file, test_data, test_labels);
}

MNIST_loader::~MNIST_loader() {
    for (double *data : train_data) {
        delete[] data;
    }
    for (double *label : train_labels) {
        delete[] label;
    }
    for (double *data : test_data) {
        delete[] data;
    }
    for (double *label : test_labels) {
        delete[] label;
    }
}

void MNIST_loader::populate_data(ifstream &file, vector<double*> &data, vector<double*> &labels) {
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
