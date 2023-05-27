#include <math.h>

/*
	Houses specific math functions used for the neural network 
*/

#ifndef NEURAL_NET_FUNCTIONS
#define NEURAL_NET_FUNCTIONS

// sigmoid function 
double sigmoid(double x)
{
	return (1/(1+exp(-x)));
}

// derivative of sigmoid used for backprop
double derivativeSigmoid(double x)
{
	return (exp(-x)/pow((1+exp(-x)),2));
}

// tanh function 
double tanh(double x)
{
	return (2.0/(1.0+exp(-2.0*x))) - 1.0;
}

// derivative of tanh used for backprop
double derivativeTanh(double x)
{
	return 1.0 - pow((2.0/(1.0+exp(-2.0*x))) - 1.0,2);
}

#endif