#ifndef LOSS_H
#define LOSS_H
#include <vector>

class Loss {
public:
    int out_size;
    double *grads;

    Loss(int _out_size);
    virtual ~Loss() { delete[] grads; };

    // Calculates loss and gradients of last layer of neural network.
    virtual double calc_loss(double *output, double *expected) = 0;
};

class MeanSquaredLoss : public Loss {
public:
    MeanSquaredLoss(int _out_size) : Loss(_out_size) {}
    
    double calc_loss(double *output, double *expected) override;
};

#endif