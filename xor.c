#include "xor.h"

int main(void)
{
    // Layers
    double a0[n0];
    double a1[n1];
    double aL[nL];

    // Biases
    double b1[n1];
    double b2[nL];

    // Weights
    double W1[n0][n1];
    double W2[n1][nL];

    // Training Set
    double X[m][n0] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    // Training Labels
    double y[m] = {0, 1, 1, 0};

    // Initialize
    initialize(W1, W2, b1, b2);

    // Train
    train(X, W1, a1, b1, W2, aL, b2, y);

    return 0;
}

void initialize(double W1[][n1], double W2[][nL], double b1[], double b2[])
{
    // Initialize W1
    for (int i = 0; i < n0; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            W1[i][j] = random();
        }
    }

    // Initialize W2
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < nL; j++)
        {
            W2[i][j] = random();
        }
    }

    // Initialize b1
    for (int i = 0; i < n1; i++)
    {
        b1[i] = random();
    }

    // Initialize b2
    for (int i = 0; i < nL; i++)
    {
        b2[i] = random();
    }
}

void train(double X[][n0], double W1[][n1], double a1[], double b1[], double W2[][nL], double aL[], double b2[], double y[])
{
    for (int i = 0; i < T; i++)
    {
        shuffle(X, y);

        for (int j = 0; j < m; j++)
        {
            // Forward Pass
            forward(X[j], W1, a1, b1, W2, aL, b2);

            // Loss
            double L = binaryCrossEntropy(aL[0], y[j]);

            // Backward Pass
            double dW2[n1][nL], dW1[n0][n1], db1[n1], db2[nL];
            backward(dW2, dW1, db1, db2, aL, y[j], a1, W2, X[j]);

            // Stochastic Gradient Descent
            stochasticGradientDescent(W1, dW1, b1, db1, W2, dW2, b2, db2);
        }
    }
}

void shuffle(double X[][n0], double y[])
{
    for (int i = 0; i < m; i++)
    {
        int j = rand() % m;

        for (int k = 0; k < n0; k++)
        {
            double temp = X[i][k];
            X[i][k] = X[j][k];
            X[j][k] = temp;
        }

        double temp = y[i];
        y[i] = y[j];
        y[j] = temp;
    }
}

void forward(double x[], double W1[][n1], double a1[], double b1[], double W2[][nL], double aL[], double b2[])
{
    // Hidden Layer
    for (int i = 0; i < n1; i++)
    {
        a1[i] = 0;

        for (int j = 0; j < n0; j++)
        {
            a1[i] += W1[j][i] * x[j];
        }

        a1[i] += b1[i];

        a1[i] = ReLU(a1[i]);
    }

    // Output Layer
    for (int i = 0; i < nL; i++)
    {
        aL[i] = 0;

        for (int j = 0; j < n1; j++)
        {
            aL[i] += W2[j][i] * a1[j];
        }

        aL[i] += b2[i];

        aL[i] = sigmoid(aL[i]);
    }
}

void backward(double dW2[][nL], double dW1[][n1], double db1[], double db2[], double aL[], double y, double a1[], double W2[][nL], double x[])
{
    // Derivative of the loss function with respect to the biases of the output layer
    for (int i = 0; i < nL; i++)
    {
        db2[i] = aL[i] - y;
    }

    // Derivative of the loss function with respect to the weights of the output layer
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < nL; j++)
        {
            dW2[i][j] = db2[j] * a1[i];
        }
    }

    // Derivative of the loss function with respect to the biases of the hidden layer
    for (int i = 0; i < n1; i++)
    {
        db1[i] = 0;

        for (int j = 0; j < nL; j++)
        {
            db1[i] += db2[j] * W2[i][j];
        }

        db1[i] *= dReLU(a1[i]);
    }

    // Derivative of the loss function with respect to the weights of the hidden layer
    for (int i = 0; i < n0; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            dW1[i][j] = db1[j] * x[i];
        }
    }
}

void stochasticGradientDescent(double W1[][n1], double dW1[][n1], double b1[], double db1[], double W2[][nL], double dW2[][nL], double b2[], double db2[])
{
    // Update W1
    for (int i = 0; i < n0; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            W1[i][j] -= ALPHA * dW1[i][j];
        }
    }

    // Update W2
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < nL; j++)
        {
            W2[i][j] -= ALPHA * dW2[i][j];
        }
    }

    // Update b1
    for (int i = 0; i < n1; i++)
    {
        b1[i] -= ALPHA * db1[i];
    }

    // Update b2
    for (int i = 0; i < nL; i++)
    {
        b2[i] -= ALPHA * db2[i];
    }
}

double random(void)
{
    return (double) rand() / RAND_MAX;
}

double ReLU(double x)
{
    return x > 0 ? x : 0;
}

double dReLU(double x)
{
    return x > 0 ? 1 : 0;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double binaryCrossEntropy(double a, double y)
{
    return -y * log(a) - (1 - y) * log(1 - a);
}