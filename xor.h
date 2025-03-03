#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Layers
#define n0 2
#define n1 2
#define nL 1

// Training Set
#define m 4

// Epochs
#define T 10000

// Learning Rate
#define ALPHA 0.1

void initialize(double W1[][n1], double W2[][nL], double b1[], double b2[]);
void train(double X[][n0], double W1[][n1], double a1[], double b1[], double W2[][nL], double aL[], double b2[], double y[]);

void shuffle(double X[][n0], double y[]);
void forward(double x[], double W1[][n1], double a1[], double b1[], double W2[][nL], double aL[], double b2[]);
void backward(double dW2[][nL], double dW1[][n1], double db1[], double db2[], double aL[], double y, double a1[], double W2[][nL], double x[]);
void stochasticGradientDescent(double W1[][n1], double dW1[][n1], double b1[], double db1[], double W2[][nL], double dW2[][nL], double b2[], double db2[]);

double random(void);
double ReLU(double x);
double dReLU(double x);
double sigmoid(double x);
double binaryCrossEntropy(double a, double y);