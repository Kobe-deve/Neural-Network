#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct node
{
	int activation;
	
	int bias;
	
};

// Neural Network data structures 

// array for the input/output layers
struct node * inputs;
struct node * output;

// array for hidden layers
struct node ** hiddenLayer;

// Training data 

int ** trainingData; // specific array of training data
int ** trainingDataLabels; // labels for training data 

int inputSize; // the size of one set of training data 
int numInDataFile; // number of sets in training data file
int labelSize; // the size of the labels 

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

	if(readFile == NULL || readOutputFile == NULL)
	{
		printf("\nERROR: UNABLE TO OPEN FILES");
		
		return;
	}
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
	
	fclose(readFile);
	free(fileReader);
}

void main()
{
	int i,j;
	readTestInputFiles("training_data/test_data");
	
	
	for(i=0;i<numInDataFile;i++)
	{
		for(j=0;j<inputSize;j++)
		{
			printf("\n%d",trainingData[i][j]);
		}
		printf("\n");
	}
	
	printf("\nOUTPUT:");
	
	for(i=0;i<numInDataFile;i++)
	{
		for(j=0;j<labelSize;j++)
		{
			printf("\n%d",trainingDataLabels[i][j]);
		}
		printf("\n");
	}
	

	// after reading file, set up the input and output layers of the network
	inputs = malloc(inputSize * sizeof(struct node));
	output = malloc(labelSize * sizeof(struct node));
}