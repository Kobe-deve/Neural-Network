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

// cost function used to show how off the network is 
double costFunction()
{
}

// function for activating a node 
void activationFunction(struct node * previousLayer, struct node * inputNode)
{
	int i,j;
	
	double z = 0;
	
	// sum of weights times activations 
	for(i=0;i<inputNode->numWeights;i++)
	{
		z += inputNode->weights[i]*previousLayer[i].activation;
	}
	
	z += inputNode->bias;
	
	inputNode->activation = sigmoid(z);
}

// prints out activations of each node in each layer
void printNodes()
{
	int i,j;
	
	printf("\n-------------------------------------");
	
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
		printf("\nLAYER %d - ",i);
		for(j=0;j<numNodesHidden;j++)
		{
			printf("%.2f ",hiddenLayer[i][j].activation);
		}
	}
	
	// output
	printf("\n\nOUTPUT\n");
	for(i=0;i<labelSize;i++)
	{
		printf("%.2f ",output[i].activation);
	}
}

void main(int argc, char *argv[])
{	
	int i,j;
	
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
	{
		inputs[i].activation = 0;
	}
	
	output = malloc(labelSize * sizeof(struct node));
	
	// initialize hidden layer 
	numHidden = 2;
	hiddenLayer = malloc(numHidden * sizeof(struct node *));
	numNodesHidden = 3;
	
	// loop through each layer 
	for(i=0;i<numHidden;i++)
	{
		hiddenLayer[i] = malloc(numNodesHidden * sizeof(struct node));
		
		// initialize weights/biases in hidden layer 
		if(i==0)
			initializeLayer(inputSize,hiddenLayer[0],numNodesHidden);
		else
			initializeLayer(numNodesHidden,hiddenLayer[i],numNodesHidden);
	
		printf("LAYER %d INITIALIZED\n",i+1);
	}
	
	// initialize weights/biases in output layer 
	initializeLayer(numNodesHidden,output,labelSize);
	printf("OUTPUT INITIALIZED");
	
	// go through all training data 
	for(i=0;i<numInDataFile;i++)
	{	
		// input training data into input nodes
		for(j=0;j<inputSize;j++)
		{
			inputs[j].activation = trainingData[i][j];
		}
		
		printNodes();
	}
	
	// free data used at the end 
	free(inputs);
	free(output);
	
	for(i=0;i<numHidden;i++)
		free(hiddenLayer[i]);
	
	free(hiddenLayer);
}