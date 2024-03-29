#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class MNIST_loader {
public:
    const int IMG_LEN = 784;
    const int N_CLASSES = 10;
    std::vector<double*> train_data;
    std::vector<double*> train_labels;
    std::vector<double*> test_data;
    std::vector<double*> test_labels;
    
    // Constructor reads in training and testing data
    MNIST_loader(std::string train_filename, std::string test_filename);
    ~MNIST_loader();

    // Populates data and labels from the file.
    void populate_data(std::ifstream &file, std::vector<double*> &data, std::vector<double*> &labels);
};

#endif