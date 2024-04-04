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
    // data is a vector of double arrays, each representing an input image.
    std::vector<double*> data;
    // labels is a vector expected outputs corresponding to the input data.
    std::vector<double*> labels;
    
    // Constructor reads in data and labels from file.
    MNIST_loader(std::string filename);
    ~MNIST_loader();
};

#endif