/*

	Created by Kobe Runnels
	5/8/2023 - 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// structure for individual nodes in the network 
struct node
{
	double activation; // activation level of the node 
	
	double bias; // holds the specific bias for the node 
	double * weights; // holds weights for previous layer 
	int numWeights; // number of weights
};

// Neural Network data structures 

// array for the input/output layers
struct node * inputs;
struct node * output;

// array for hidden layers
struct node ** hiddenLayer;

// the number of hidden layers in the network 
int numHidden; 

// how many nodes are in each hidden layer (TODO: change to array probably)
int numNodesHidden;

// Training data 

int ** trainingData; // specific array of training data
int ** trainingDataLabels; // labels for training data 

int inputSize; // the size of one set of training data 
int numInDataFile; // number of sets in training data file
int labelSize; // the size of the labels 

double learningRate = 0.1; // the learning rate of the network 

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

// for error handling 
void throwError(char * errorMessage)
{
	printf("ERROR: %s",errorMessage);
	exit(1);
}

// reads the input file 
void readTestInputFiles(char * fileName)
{
	FILE *readFile;
	FILE *readOutputFile;
	char * fileReader = malloc(128 * sizeof(char)); 
	int i,j;

	char inputFile[128] = "";
	char outputFile[128] = "";
	
	// set up reading training data input file 
	strcat(inputFile,fileName);
	strcat(inputFile,"_input.txt");
	
	readFile = fopen(inputFile,"r");
	
	// set up reading training label output file 
	strcat(outputFile,fileName);
	strcat(outputFile,"_output.txt");
	
	readOutputFile = fopen(outputFile,"r");
	
	// start reading the files 
	if(readFile == NULL || readOutputFile == NULL)
		throwError("UNABLE TO OPEN FILES");
	else
	{
		// get input file set size and number of sets for input to network size 
		fscanf(readFile,"%s",fileReader);
		numInDataFile = atoi(fileReader);
		
		fscanf(readFile,"%s",fileReader);
		inputSize = atoi(fileReader);
		
		// allocate training array
		trainingData = malloc(numInDataFile * sizeof(int **));
		for(i=0;i<numInDataFile;i++)
		{	
			trainingData[i] = malloc(inputSize * sizeof(int));
			for(j=0;j<inputSize;j++)
				trainingData[i][j] = 0;
		}	
		
		// read and obtain training data 
		for(i=0;i<numInDataFile;i++)
		{
			// read specific set
			for(j=0;j<inputSize;j++)
			{
				fscanf(readFile,"%s",fileReader);
				trainingData[i][j] = atoi(fileReader);
			}
		}
		
		// read training label file
		
		fscanf(readOutputFile,"%s",fileReader);
		labelSize = atoi(fileReader);
		
		// allocate training array
		trainingDataLabels = malloc(numInDataFile * sizeof(int **));
		for(i=0;i<numInDataFile;i++)
		{	
			trainingDataLabels[i] = malloc(labelSize * sizeof(int));
			for(j=0;j<labelSize;j++)
				trainingDataLabels[i][j] = 0;
		}	
		
		// read and obtain training data 
		for(i=0;i<numInDataFile;i++)
		{
			// read specific set
			for(j=0;j<labelSize;j++)
			{
				fscanf(readOutputFile,"%s",fileReader);
				trainingDataLabels[i][j] = atoi(fileReader);
			}
		}
	}
	
	// close files and free file reader used 
	fclose(readFile);
	fclose(readOutputFile);
	free(fileReader);
}

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

// cost function used to show how much error the network has with mean squared error  
double costFunction(int trainingDataSet)
{
	int i;
	
	double sum = 0;
	
	// go through outputs and subtract their activations with the expected label output 
	for(i=0;i<labelSize;i++)
	{
		int active = 0;
		
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
	inputNode->activation = fmin(1,fmax(sigmoid(z),0.000000000000000000001)); // check used so values aren't NaN
	
	if(inputNode->activation < 0)
		inputNode->activation = 0;
}

// prints out activations of each node in each layer
void printNodes()
{
	int i,j;
	
	printf("\n\n-------------------------------------");
	
	// input 
	printf("\nINPUT:\n");
	
	for(i=0;i<inputSize;i++)
	{
		printf("%.2f ",inputs[i].activation);
	}
	
	printf("\n\nHIDDEN");
	
	// hidden 
	for(i=0;i<numHidden;i++)
	{
		printf("\nLAYER %d - (NUM WEIGHTS BEFORE: %d): ",i,hiddenLayer[i][0].numWeights);
		for(j=0;j<numNodesHidden;j++)
		{
			printf("%f ",hiddenLayer[i][j].activation);
		}
	}
	
	// output
	printf("\n\nOUTPUT (NUM WEIGHTS BEFORE: %d):\n",output[0].numWeights);
	for(i=0;i<labelSize;i++)
	{
		printf("%f ",output[i].activation);
	}
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
			else // if on the other hidden layers, get the previous layer 
			{
				for(j=0;j<numNodesHidden;j++)
				{
					errorHidden += deltaHidden[i-1][j] * hiddenLayer[i][z].weights[j];
				}
				deltaHidden[i][z] = errorHidden*derivativeSigmoid(hiddenLayer[i][z].activation);
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

void main(int argc, char *argv[])
{	
	int i,j,z,x;
	
	// set the seed for randomization (when initializing)
	srand((unsigned)time(NULL));
	
	// command line input 
	if( argc == 2 ) 
		readTestInputFiles(argv[1]);
	else
		readTestInputFiles("training_data/test_data");

	// after reading file, set up the input and output layers of the network
	inputs = malloc(inputSize * sizeof(struct node));
	
	// set default activation of input nodes to 0
	for(i=0;i<inputSize;i++)
		inputs[i].activation = 0;
	
	printf("\nINPUT LAYER INITIALIZED");
	
	output = malloc(labelSize * sizeof(struct node));
	
	// initialize hidden layer 
	numHidden = 2;
	hiddenLayer = malloc(numHidden * sizeof(struct node *));
	numNodesHidden = 3;
	
	// initialize hidden layer before input 
	hiddenLayer[0] = malloc(numNodesHidden * sizeof(struct node));
	initializeLayer(inputSize,hiddenLayer[0],numNodesHidden);
	printf("\nHIDDEN LAYER 1 INITIALIZED\n",i+1);
		
	// loop through each layer for initialization 
	for(i=1;i<numHidden;i++)
	{
		hiddenLayer[i] = malloc(numNodesHidden * sizeof(struct node));
		
		// initialize weights/biases in hidden layer 
		initializeLayer(numNodesHidden,hiddenLayer[i],numNodesHidden);
	
		printf("HIDDEN LAYER %d INITIALIZED\n",i+1);
	}
	
	// initialize weights/biases in output layer 
	initializeLayer(numNodesHidden,output,labelSize);
	printf("OUTPUT INITIALIZED");

	// loop through training 1000 times
	for(x=0;x<10000;x++)
	{
		// go through all training data 
		for(i=0;i<numInDataFile;i++)
		{	
			// input training data into input nodes
			for(j=0;j<inputSize;j++)
			{
				inputs[j].activation = trainingData[i][j];
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
			
			// activate output layer 
			for(z=0;z<labelSize;z++)
			{
				activationFunction(hiddenLayer[numHidden-1], &output[z]);	
			}
			
			// display activation of all nodes and the cost 
			//printNodes();
			
			// back propagation
			backPropagation(i);
			
			int valid = 1;
			
			// check if the result is true 
			for(z=0;z<labelSize;z++)
			{
				if(output[z].activation >= 0.5 && trainingDataLabels[i][z] == 0)
				{
					valid = 0;
					break;
				}
			}
			
			if(x==0 || x== 9999)
			{
				if(valid == 1)
					printf("\nTRUE");
				else
					printf("\nFALSE");
		
			}
		}
	}
		
	// free data used at the end 
	free(inputs);
	free(output);
	
	for(i=0;i<numHidden;i++)
		free(hiddenLayer[i]);
	
	free(hiddenLayer);
}