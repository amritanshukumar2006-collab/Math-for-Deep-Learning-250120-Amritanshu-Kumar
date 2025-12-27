// Making a Neural Network in C Language, I had tried to make the Neural Network in C rather than c++
// Here we have made the following Specified functions: -
// 1. Loss calculator
// 2. Gradient descent
// 3. Training loop
// 4. Testing loop

#include <stdio.h>
#include <math.h>

// Taking the Activation Function as designed erlier

double beta = 0.5; // Taking the value of Beta as 0.5
double activation(double x) {
    if (x < -1.0){
        return beta * x - (1.0 - beta);
    }
    else if (x > 1.0){
        return beta * x + (1.0 - beta);
    }
    else{
        return x;
    }
}

// Function finding the Derivative of the Activation Function

double activation_derivative(double x) {
    if (x < -1.0 || x > 1.0)// Here we are doing only one Comparision Operation 
        return beta;        // Just the Comparision Operation decides the derivative make the Computation fast and inexperience
    else
        return 1.0;
}

// Creating a Forward Pass Function for the Activation Function 

double forward(double x, double w, double b) {
    double z = w * x + b; // w is the weight and b is the bias for the respective weights
    return activation(z); // Here, z is the preactivation of the Neural Network
}

// Creating a Cost Function

double loss(double y, double y_prediction) {
    double difference = y - y_prediction; // Here, difference is the loss
    return 0.5 * difference * difference; // Putting the Value of the Loss into the Cost Function
}

// Creating a Function for the Computation of the Gradient

void compute_gradients(double x, double y,
                       double w, double b,
                       double *dw, double *db) {

    double z = w * x + b; // Here, z is the preactivation for the Neural Network
    double y_prediction = activation(z); // The activation function is predicting the value of y
    double error = y_prediction - y; // This is the loss amount of the information
    double dz = activation_derivative(z); // Calculating the Gradient of the Activation function wrt the predicted value of y
    
    // The lost function in L = 1/2(y_predicted - y)^2

    *dw = error * dz * x; // Applying the Chain Rule and finding the derivative of lost function wrt weights
    *db = error * dz; // Applying the Chain Rule and finding the derivative of lost function wrt biases
}

// Creating a function for the Upgradation of the Gradient Descent

void gradient_descent(double *w, double *b,
                      double dw, double db,
                      double lr) {

    *w -= lr * dw; // Upgrading the value of the weights
    *b -= lr * db; // Upgrading the value of the biases
}

// Creating a Training Loop Function
// This is Training loop similar to the Python Training Loop
void train(double *w, double *b,
           double *x, double *y,
           int n, int epochs, double lr) {
// Here, we are using the Stochastic Gradient Descent model
    for (int e = 0; e < epochs; e++) {
        double total_loss = 0.0;

        for (int i = 0; i < n; i++) {
            double y_prediction = forward(x[i], *w, *b);
            total_loss += loss(y[i], y_prediction);

            double dw, db;
            compute_gradients(x[i], y[i], *w, *b, &dw, &db);
            gradient_descent(w, b, dw, db, lr);
        }

        printf("Epoch %d | Loss = %.6f\n", e, total_loss / n);
    }
}

// Creating a Testing Loop Function and finding out the prediction value as well as the loss of the original value
// Finally printing all those values

void test(double w, double b,
          double *x_test, double *y_test, int n) {
// Here we are printing out the predicted value of y by our Neural Network 
    printf("\nTesting:\n");
    for (int i = 0; i < n; i++) {
        double y_prediction = forward(x_test[i], w, b);
        printf("x = %.2f, y = %.2f, y prediction = %.2f\n",
               x_test[i], y_test[i], y_prediction);
    }
}

// Fuction for finding out the Accuracy 

double compute_accuracy(double w, double b,
                        double *x, double *y,
                        int n, double tolerance) {

    int correct = 0;

    for (int i = 0; i < n; i++) {
        double z = w * x[i] + b;
        double y_pred = activation(z);

        if (fabs(y_pred - y[i]) <= tolerance) { // Taking the tolerence as 0.8, means that a deviation of +- 0.8 is considered as correct
            correct++;
        }
    }

    return (double)correct / n;
}

int main() {
    // The Neural Network is predicting the value of y with taking x as the input set
    double x_train[] = {1, 2, 3, 4}; // Taking the x - axis Training Dataset
    double y_train[] = {2, 4, 6, 8}; // Taking the y - axis Training Dataset
    int n_train = 4; // Number of Training data

    double w = 0.1;
    double b = 0.0;

    train(&w, &b, x_train, y_train, n_train, 200, 0.01); // Taking the Learning Rate as 0.01 and the epoch 200 for better prediction value

    double x_test[] = {5, 6};
    double y_test[] = {10, 12};

    test(w, b, x_test, y_test, 2); // 2 is the Number of Testing data - 5 and 6
    double tolerance = 0.8;
    double accuracy = compute_accuracy(w, b, x_test, y_test, 2, tolerance); // Printing out the Accuracy of the Neural Network

    printf("\nAccuracy = %.2f%%\n", accuracy * 100);

    return 0;
}
