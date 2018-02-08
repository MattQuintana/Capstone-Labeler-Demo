package capstone_demo;

import java.util.Arrays;

public class NeuralNetwork {
	
	int number_nodes;
	double[][] w_mean_array;
	double[] input_layer;
	
	// Create the size of the net to 
	public NeuralNetwork(double[] input_array) {
		this.w_mean_array = new double[input_array.length][4];
		initializeNet();
	}
	
	// Initialize the net to random value
	private void initializeNet() {
		// Fill matrix with random weights between 0 and 1
        for(int i = 0; i < w_mean_array.length; i++) {
        	for(int j = 0; j < w_mean_array[i].length; j++) {
        		w_mean_array[i][j] = Math.random();
        	}
        }
	}
	
	private static double nonlin(double x) {
    	return 1/(1 + Math.exp(-x));
    }
	
	// Process an input array of values
	public int processInput(double[] input_array, int goal_output) {
        int guess_value = -1;
        
        while (guess_value != goal_output) {
        	// Output layer to hold final proficiency guesses
        	double[] output_layer = new double[4];
            
            // Do the matrix multiplication
        	// Basically multiply all of the mean values retrieved from the attributes 
        	// by the randomized weights in the matrix. 
            for(int matx_col = 0; matx_col < w_mean_array[0].length; matx_col++) {
            	double col_sum = 0;
            	for(int matx_row = 0; matx_row < w_mean_array.length; matx_row++) {
            		col_sum += input_array[matx_row] * w_mean_array[matx_row][matx_col];
            	}
            	// Put all of those multiplications into the corresponding output layer node
            	output_layer[matx_col] = col_sum;
            }
        
        
	        // Normalize the value to be between 0 and 1
	        for(int i = 0; i < output_layer.length; i++) {
	        	output_layer[i] = nonlin(output_layer[i]);
	        }
	        
	        System.out.println(Arrays.toString(output_layer));
	        
	        // Identify the max value in the output layer 
	        // Find which index it corresponds to aka which proficiency level 
	        double max = 0.0;
	        int max_index = 0;
	        for(int max_i = 0; max_i < output_layer.length; max_i++) {
	        	if (output_layer[max_i] > max) {
	        		max = output_layer[max_i];
	        		max_index = max_i;
	        	}
	        }
	        System.out.println(max_index);
	        guess_value = max_index;
	        
	        System.out.println("Actual proficiency: CAE");
	        // Display guess to terminal
	        switch(max_index) {
	        	case 0:
	        		System.out.println("Guess CAE proficiency.\n");
	        		break;
	        	case 1: 
	        		System.out.println("Guess CDE proficiency.\n");
	        		break;
	        	case 2:
	        		System.out.println("Guess FCE proficiency.\n");
	        		break;
	        	case 3:
	        		System.out.println("Guess PET proficiency.\n");
	        		break;
	        }
	        
	        
	        // If we've made a wrong guess
	        if (guess_value != goal_output) {
	        	// Adjust the weights
	        	// Lower them by some kind of factor based on the difference between the actual and guess score. 
	        	// Lower the weights associated with the wrong guess 
	        	for (int matx_row = 0; matx_row < w_mean_array.length; matx_row++) {
	        		for (int matx_col = 0; matx_col < w_mean_array[matx_row].length; matx_col++) {
	        			if (matx_col == max_index) {
	        				w_mean_array[matx_row][max_index] *= 1 - nonlin(Math.abs(goal_output-guess_value));
	        				w_mean_array[matx_row][max_index] = nonlin(w_mean_array[matx_row][matx_col]);
	        			}
	        			else {
	        				// Boost the weights that are not associated with the wrong guess
	        				w_mean_array[matx_row][matx_col] *= 1 + nonlin(Math.abs(goal_output-guess_value));
	        				w_mean_array[matx_row][matx_col] = nonlin(w_mean_array[matx_row][matx_col]);
	        			}
	        		}
	        	}
	        }	        
        }
        return guess_value;
	}
	
}
