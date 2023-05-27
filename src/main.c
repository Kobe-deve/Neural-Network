/*

	Created by Kobe Runnels
	5/8/2023 - 
*/


#include "lib/neural_net.h"
#include "lib/neural_net_functions.h"
#include "lib/neural_net_file_reading.h"
#include <time.h>

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

// prints weights in the network
void printWeightsAndBiases()
{
	int i,j,z;
	
	printf("\n\n-------------------------------------");
	
	printf("\n\nHIDDEN");
	
	// hidden 
	for(i=0;i<numHidden;i++)
	{
		printf("\nLAYER %d WEIGHTS/BIASES:",i+1);
		for(j=0;j<numNodesHidden;j++)
		{
			printf("\nBIAS - %.2f\n",hiddenLayer[i][j].bias);
			for(z=0;z<hiddenLayer[i][j].numWeights;z++)
				printf("\n%f ",hiddenLayer[i][j].weights[z]);
			printf("\n");
		}
	}
	
	printf("\n");
	
	// output
	printf("\nOUTPUT WEIGHTS/BIASES\n");
	for(i=0;i<labelSize;i++)
	{
		for(j=0;j<numNodesHidden;j++)
		{
			printf("\nBIAS - %.2f",output[i].bias);
			for(z=0;z<output[i].numWeights;z++)
				printf("\n%f ",output[i].weights[z]);
			printf("\n");
		}
	}
}

void main(int argc, char *argv[])
{	
	int i,j,z,x;
	
	/* initialization of the program/network */
	
	// set the seed for randomization (when initializing)
	srand((unsigned)time(NULL));
	
	// set learning rate and number of iterations through training data 
	numTestIterations = 10000;
	learningRate = 0.001;
	
	// initialize cost progression array 
	costProgression = malloc(numTestIterations * sizeof(double));
	
	// command line input for training data 
	if( argc == 2 ) 
		readTrainingInputFiles(argv[1]);
	else
		readTrainingInputFiles("training_data/test_data");

	// after reading file, set up the input and output layers of the network
	inputs = malloc(inputSize * sizeof(struct node));
	
	// set default activation of input nodes to 0
	for(i=0;i<inputSize;i++)
		inputs[i].activation = 0;
	
	printf("\nINPUT LAYER INITIALIZED");
	
	output = malloc(labelSize * sizeof(struct node));
	
	// initialize hidden layer array
	numHidden = 1;
	hiddenLayer = malloc(numHidden * sizeof(struct node *));
	numNodesHidden = 6;
	
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

	/* Training the network - iterating through the test data and using backprop to learn */

	// loop through training data (numTestIterations) number of times
	for(x=0;x<numTestIterations;x++)
	{
		double cost = 0.0; // used for getting cost for one run of all training data
		
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
			
			// back propagate/learn after going through the training data set 
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
			
			cost += costFunction(i);
		}
		
		//printNodes();
		costProgression[x] = cost;
	}
	
	/* Testing network - giving it test data and viewing results */
	
	// display cost progression difference
	printf("\n\nCOST PROGRESSION (Last test run - First test) - %.10f",costProgression[numTestIterations-1]-costProgression[0]);
	
	// get testing data from files
	if( argc == 3 ) 
		readTestInputFiles(argv[2]);
	else
		readTestInputFiles("testing_data/test_data");

	// go through testing data and output the network's output vs the expected
	for(i=0;i<numInTestDataFile;i++)
		neuralNetwork(testingData[i],testingDataLabels[i]);
	
	//printWeightsAndBiases();
	
	/* End of program - Freeing up memory */
	
	// free data used at the end 
	free(inputs);
	free(output);
	
	// free cost progression array 
	free(costProgression);
	
	// free hidden layer array 
	for(i=0;i<numHidden;i++)
		free(hiddenLayer[i]);
	
	free(hiddenLayer);
	
	/*
	Example code from https://github.com/takafumihoriuchi/MNIST_for_C for reading MNIST in C, will be used for testing 
	
	// call to store mnist in array
    load_mnist();

    // print pixels of first data in test dataset
    int i;
    for (i=0; i<784; i++) {
        printf("%1.1f ", test_image[0][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }

    // print first label in test dataset
    printf("label: %d\n", test_label[0]);

    // save image of first data in test dataset as .pgm file
    save_mnist_pgm(test_image, 0);

    // show all pixels and labels in test dataset
    print_mnist_pixel(test_image, NUM_TEST);
    print_mnist_label(test_label, NUM_TEST);
	*/
}