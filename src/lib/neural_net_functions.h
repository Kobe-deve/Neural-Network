#include <math.h>

/*
	Houses specific math functions used for the neural network 
*/

#ifndef NEURAL_NET_FUNCTIONS
#define NEURAL_NET_FUNCTIONS

// sigmoid function used 
double sigmoid(double x)
{
	return (1/(1+exp(-x)));
}

// derivative of sigmoid used for backprop
double derivativeSigmoid(double x)
{
	return (exp(-x)/pow((1+exp(-x)),2));
}

#endif