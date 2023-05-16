# C Neural Network

Developed By Kobe Runnels

# Running
	main (txt file name/path from exe for training data) (txt file name/path from exe for testing data)
	
# Compiling
	gcc main.c -o main

# File input format:
	Files
		(Training data)
		-training_data/(File Name)_input.txt
			-First line: number of data sets in both training data files 
			-Second line: size of input (input layer nodes to set)
			-Specifically labels which nodes to activate for input 
		-training_data/(File Name)_output.txt
			-First line: Size of output labels (output layer nodes to set)
			-Specifically labels which nodes should be active on output 
			-This is used for the cost function but in the future this will be replaced with regular
			 labels with a given range of what outputs should be expected
		
		(Testing data)
		-testing_data/(File Name)_input.txt
			-First line: number of data sets in both testing data files 
			-Specifically labels which nodes to activate for input 
		-testing_data/(File Name)_output.txt
			-Specifically labels which nodes should be active on output 
		
# Process
	-Take in training data (inputs/labels)
	-Train on data and develop cost function
	-Calculate cost slope from all training sets - gradient descent/downward hill climbing  
		-Compute gradient of cost function 
	-Step in negative gradient direction 
		-Change weights/biases 
				
# Resources/Inspiration
- http://neuralnetworksanddeeplearning.com/
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547