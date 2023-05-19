#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_net_functions.h"

#ifndef NEURAL_NET
#define NEURAL_NET

// structure for individual nodes in the network 
struct node
{
	double activation; // activation level of the node 
	
	double bias; // holds the specific bias for the node 
	double * weights; // holds weights for previous layer 
	int numWeights; // number of weights
};

// the min/max activations a node can have 
#define MIN_NODE_ACTIVATION 0
#define MAX_NODE_ACTIVATION 1

// Neural Network data structures 

double * costProgression; // holds an array of the cost progression for every run  

// array for the input/output layers
struct node * inputs;
struct node * output;

int numTestIterations; // number of times the training data is run through 

// array for hidden layers
struct node ** hiddenLayer;

// the number of hidden layers in the network 
int numHidden; 

// how many nodes are in each hidden layer (TODO: change to array probably)
int numNodesHidden;

// Training data 

int ** trainingData; // specific array of training data
int ** trainingDataLabels; // labels for training data 

int ** testingData; // specific array for testing data 
int ** testingDataLabels; // labels for testing data 
int numInTestDataFile; // number of data sets in testing data file 

int inputSize; // the size of one set of training data 
int numInDataFile; // number of sets in training data file
int labelSize; // the size of the labels 

double learningRate; // the learning rate of the network 

// initialize a specific layer 
void initializeLayer(int prevLayerSize, struct node * layer, int layerSize)
{
	int i, j;
	
	// initialize nodes in layer with random value between 0 and 1 and set up weights 
	for(i=0;i<layerSize;i++)
	{
		layer[i].activation = 0;
		layer[i].bias = ((double)rand())/((double)RAND_MAX);
		
		// set up weights and initialize them 
		layer[i].weights = malloc(prevLayerSize * sizeof(double));
		layer[i].numWeights = prevLayerSize;
		
		for(j=0;j<prevLayerSize;j++)
			layer[i].weights[j] = ((double)rand())/((double)RAND_MAX);
	}
}

// cost function used to show how much error the network has with mean squared error for one training data set  
double costFunction(int trainingDataSet)
{
	int i;
	
	double sum = 0.0;
	
	// go through outputs and subtract their activations with the expected label output 
	for(i=0;i<labelSize;i++)
	{
		sum += pow(output[i].activation - trainingDataLabels[trainingDataSet][i],2);
	}
	
	return sum/labelSize;
}

// function for activating a node 
void activationFunction(struct node * previousLayer, struct node * inputNode)
{
	int i,j;
	
	double z = 0; // weighted input 
	
	// sum of weights times activations 
	for(i=0;i<inputNode->numWeights;i++)
	{
		z += inputNode->weights[i]*previousLayer[i].activation;
	}
	
	// add the bias 
	z += inputNode->bias;
	
	// set activation 
	inputNode->activation = fmin(MAX_NODE_ACTIVATION,fmax(sigmoid(z),MIN_NODE_ACTIVATION)); // check used so values aren't NaN
	
	if(inputNode->activation < 0)
		inputNode->activation = 0;
}

/*
   derivative of cost with respect to weight
   = activation of node in last layer * derivative sigmoid(z) * 2 * (activation - expected)
   
   derivative of cost with respect to bias
   = 1 * derivative sigmoid(z) * 2 * (activation - expected)
   
   derivative of cost with respect to activation of last node
   = (sum of nodes through layer) (weight * derivative sigmoid(z) * 2 * (activation - expected))
*/
// takes in a specific training set to determine back propagation with updating weights
// TODO decreasing biases/weights?
void backPropagation(int trainingSet)
{
	// iteration loop variables 
	int i,j,z;
	
	// arrays used for setting up the weights with error calculation 
	double * deltaOutput = malloc(labelSize * sizeof(double));
	double ** deltaHidden = malloc(numHidden * sizeof(double *));
	
	// allocate delta hidden 
	for(i=0;i<numHidden;i++)
		deltaHidden[i] = malloc(numNodesHidden * sizeof(double));
	
	// output layer calculations
	for(i=0;i<labelSize;i++)
	{
		// calculate the error of the activation of the output - the expected activation 
		double error = output[i].activation - trainingDataLabels[trainingSet][i];
		
		// set up delta output calculation for setting weights
		deltaOutput[i] = error*derivativeSigmoid(output[i].activation);
	}
	
	// hidden layers calculations
	if(numHidden == 1)
	{
		for(z=0;z<numNodesHidden;z++)
		{
			double errorHidden = 0.0; // error in the current hidden layer
			
			// get the output layer info 
			for(j=0;j<labelSize;j++)
			{
				errorHidden += deltaOutput[j] * hiddenLayer[0][z].weights[j];
			}
			
			deltaHidden[0][z] = errorHidden*derivativeSigmoid(hiddenLayer[0][z].activation);
		}
	}
	else
	{
		for(i=numHidden-1;i>=1;i--)
		{
			double errorHidden = 0.0; // error in the current hidden layer
			
			for(z=0;z<numNodesHidden;z++)
			{
				// if this is the hidden layer before the output zone, get the output layer info 
				if(i == numHidden-1)
				{
					for(j=0;j<labelSize;j++)
					{
						errorHidden += deltaOutput[j] * hiddenLayer[i][z].weights[j];
					}
				
					deltaHidden[i][z] = errorHidden*derivativeSigmoid(hiddenLayer[i][z].activation);
				
				}
				else if(i >= 1) // if on the other hidden layers, get the previous layer 
				{
					for(j=0;j<numNodesHidden;j++)
					{
						errorHidden += deltaHidden[i-1][j] * hiddenLayer[i][z].weights[j];
					}
					deltaHidden[i][z] = errorHidden*derivativeSigmoid(hiddenLayer[i][z].activation);
				}
				
			}
		}
	}
	
	// set output layer weights/biases
	for(i=0;i<labelSize;i++)
	{
		output[i].bias += deltaOutput[i]*learningRate; // set the bias for the specific node 
		for(j=0;j<numNodesHidden;j++)
		{
			output[i].weights[j] += hiddenLayer[numHidden-1][z].activation*deltaHidden[numHidden-1][j]*learningRate;
		}
	}
	
	// set hidden layer weights/biases
	if(numHidden != 1)
	{
		for(i=numHidden-1;i>=1;i--)
		{
			for(j=0;j<numNodesHidden;j++)
			{
				hiddenLayer[i][j].bias += deltaHidden[i][j]*learningRate; // set the bias for the specific node 
				for(z=0;z<numNodesHidden;z++) // set weights 
				{
					hiddenLayer[i][j].weights[z] += hiddenLayer[i-1][z].activation*deltaHidden[i-1][j]*learningRate;
				}
			}
		}
	}
	
	// set hidden layer (before input layer) weights/biases
	for(j=0;j<numNodesHidden;j++)
	{
		hiddenLayer[0][j].bias += deltaHidden[0][j]*learningRate; // set the bias for the specific node	
		for(i=0;i<inputSize;i++) // set weights based on input data 
		{
			hiddenLayer[0][j].weights[i] += deltaHidden[0][j]*trainingData[trainingSet][i]*learningRate;
		}
	}
	
	
	// free the data used 
	free(deltaOutput);
	
	for(i=0;i<numHidden;i++)
		free(deltaHidden[i]);
	free(deltaHidden);
}

// takes in an activation array of size (inputSize) and prints out node output
void neuralNetwork(int * activationArray, int * expectedOutputArray)
{
	int i,j,z;
	
	// input training data into input nodes
	for(j=0;j<inputSize;j++)
	{
		inputs[j].activation = (double)activationArray[j];
	}
			
	// go through hidden layers and activate nodes
	for(z=0;z<numHidden;z++)
	{
		for(j=0;j<numNodesHidden;j++)
		{
			if(z==0)
				activationFunction(inputs, &hiddenLayer[z][j]);
			else
				activationFunction(hiddenLayer[z-1], &hiddenLayer[z][j]);	
		}
	}
		
	printf("\nOUTPUT:	 EXPECTED:");
	// activate output layer 
	for(z=0;z<labelSize;z++)
	{
		activationFunction(hiddenLayer[numHidden-1], &output[z]);	
		printf("\n%.2f		 %d",output[z].activation,expectedOutputArray[z]);
	}
}


#endif
