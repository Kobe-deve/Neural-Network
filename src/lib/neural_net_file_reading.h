#include "neural_net.h"

#ifndef NEURAL_NET_FILE_READING
#define NEURAL_NET_FILE_READING

// for error handling 
void throwError(char * errorMessage)
{
	printf("ERROR: %s",errorMessage);
	exit(1);
}

// reads the training data input file 
void readTrainingInputFiles(char * fileName)
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

// reads the testing data input file 
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
		// get number of test data sets 
		fscanf(readFile,"%s",fileReader);
		numInTestDataFile = atoi(fileReader);
		
		// allocate training array
		testingData = malloc(numInTestDataFile * sizeof(int **));
		for(i=0;i<numInTestDataFile;i++)
		{	
			testingData[i] = malloc(inputSize * sizeof(int));
			for(j=0;j<inputSize;j++)
				testingData[i][j] = 0;
		}	
		
		// read and obtain training data 
		for(i=0;i<numInTestDataFile;i++)
		{
			// read specific set
			for(j=0;j<inputSize;j++)
			{
				fscanf(readFile,"%s",fileReader);
				testingData[i][j] = atoi(fileReader);
			}
		}
		
		// read testing label file

		// allocate training array
		testingDataLabels = malloc(numInTestDataFile * sizeof(int **));
		for(i=0;i<numInTestDataFile;i++)
		{	
			testingDataLabels[i] = malloc(labelSize * sizeof(int));
			for(j=0;j<labelSize;j++)
				testingDataLabels[i][j] = 0;
		}	
		
		// read and obtain training data 
		for(i=0;i<numInTestDataFile;i++)
		{
			// read specific set
			for(j=0;j<labelSize;j++)
			{
				fscanf(readOutputFile,"%s",fileReader);
				testingDataLabels[i][j] = atoi(fileReader);
			}
		}
	}
	
	// close files and free file reader used 
	fclose(readFile);
	fclose(readOutputFile);
	free(fileReader);
}


#endif 