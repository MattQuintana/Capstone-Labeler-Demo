package capstone_demo;

import java.io.*;
import java.util.Arrays;
import java.util.*;

public class NeuralNetwork {
	
	int number_nodes;
	double[][] w_mean_array;
	double[][] hidden_layer; 
	double[] input_layer;
	
	numJava nj = new numJava();
	
	// Create the size of the net to 
	public NeuralNetwork(double[] input_array) {
		this.w_mean_array = new double[input_array.length][4];
		this.hidden_layer = new double[4][4];
		initializeNet(w_mean_array);
		initializeNet(hidden_layer);
	}
	
	// Initialize the net to random value
	private void initializeNet(double[][] array) {
		// Fill matrix with random weights between 0 and 1
        for(int i = 0; i < array.length; i++) {
        	for(int j = 0; j < array[i].length; j++) {
        		array[i][j] = Math.random() * 2 - 1;
        	}
        }
	}
	
	// Process an input array of values
	public int processInput(double[] input_array, int goal_output) {
        int guess_value = -1;
        
        double[] goal = new double[4];
        goal[goal_output] = 1.0;
        
        /*
        while (guess_value != goal_output) {
        	
	               
        }*/
        
    	// Output layer to hold final proficiency guesses
        double[] first_layer = nj.nonlinVector(nj.dot1D(input_array, w_mean_array), false);
        
        // System.out.println(Arrays.toString(first_layer));
        double[] output_layer = nj.nonlinVector(nj.dot1D(first_layer, hidden_layer), false);
        
        // Show what the output layer values are
        System.out.println(Arrays.toString(output_layer));
        
        // Make the guess to show to the console
        guess_value = makeGuess(output_layer, goal_output);
        
        // Begin the adjustment calculations
        
        // First calculate how much error the guess made
        double[] output_error = nj.subtract(goal, output_layer);
        // 
        double[] output_delta = nj.multiply(output_error, nj.nonlinVector(output_layer, true));
        // System.out.println(Arrays.toString(output_delta));
        
        /*
        for (double[] row : hidden_layer)
        {
        	System.out.println(Arrays.toString(row));
        }*/
       
        
        double[] first_error = nj.dot1D(output_delta, nj.transpose(hidden_layer));
        // System.out.println(Arrays.toString(first_error));
        
        double[] first_delta = nj.multiply(first_error, nj.nonlinVector(first_layer, true));
        // System.out.println(Arrays.toString(first_delta));
        
        // Get the adjustment of weights to be made to the second layer
        double[][] hidden_adjustment = nj.dot1D_rev(nj.transpose1D(first_layer), output_delta);
        hidden_layer = nj.add2D(hidden_layer, hidden_adjustment);
        
        // System.out.println(Arrays.toString(input_array));
        // System.out.println(Arrays.toString(first_delta));
        
        // Get the adjustment of weights to be made to the first layer
        double[][] first_adjustment = nj.dot1D_rev(nj.transpose1D(input_array), first_delta);
        //System.out.println(Arrays.toString(first_adjustment));
        w_mean_array = nj.add2D(w_mean_array, first_adjustment);

        return guess_value;
	}
	
	// Test a single input to see how the network guesses
	public int testInput(double[] input_array, int goal_output)
	{
    	// Output layer to hold final proficiency guesses
        double[] first_layer = nj.nonlinVector(nj.dot1D(input_array, w_mean_array), false);
        
        // System.out.println(Arrays.toString(first_layer));
        double[] output_layer = nj.nonlinVector(nj.dot1D(first_layer, hidden_layer), false);
        //System.out.println(Arrays.toString(output_layer));
        
        int guess_value = makeGuess(output_layer, goal_output);
        return guess_value;
	}

	// Display guess values in human readable
	public int makeGuess(double[] output_layer, int goal_output)
	{
		int guess_value;
		// Identify the max value in the output layer 
        // Find which index it corresponds to aka which proficiency level 
        double min = 1.0;
        int min_index = 0;
        for(int min_i = 0; min_i < output_layer.length; min_i++) {
        	if (output_layer[min_i]  < min) {
        		min = output_layer[min_i];
        		min_index = min_i;
        	}
        }
        //System.out.println(max_index);
        guess_value = min_index;
        
        switch(goal_output) {
        	case 0:
        		System.out.println("Actual proficiency: CAE");
        		break;
        	case 1: 
        		System.out.println("Actual proficiency: CDE");
        		break;
        	case 2:
        		System.out.println("Actual proficiency: FCE");
        		break;
        	case 3: 
        		System.out.println("Actual proficiency: PET");
        		break;
        }
        // Display guess to terminal
        switch(min_index) {
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
        
        return guess_value;
	}
	
	// Save the weights to a file
	private void saveNet(double[][] layer_1, double[][] layer_2)
	{
		File weights_file = new File("net_weights.dat");
		try 
		{
			DataOutputStream outstream = new DataOutputStream(new FileOutputStream(weights_file));
			for(double[] row: layer_1)
			{
				for (double column : row)
				{
					outstream.writeDouble(column);
				}
			}
			for(double[] row: layer_2)
			{
				for (double column : row)
				{
					outstream.writeDouble(column);
				}
			}
		}
		catch (Exception e)
		{
			System.out.println(e);
		}	
	}
	
	private void loadWeights(File weights_file)
	{
		try 
		{
			// BufferedReader instream = new(BufferedReader(new FileInputStream(weights_file));
			
			Scanner scan = new Scanner(weights_file);
			
			for (int row = 0; row < w_mean_array.length; row++)
			{
				for (int column = 0; column < w_mean_array[0].length; column++)
				{
					w_mean_array[row][column] = scan.nextDouble();
				}
			}
			
			for (int row = 0; row < hidden_layer.length; row++)
			{
				for (int column = 0; column < hidden_layer[0].length; column++)
				{
					hidden_layer[row][column] = scan.nextDouble();
				}
			}
		}
		catch (Exception e)
		{
			//System.out.println(e);
		}
			
		
	}
	
	// Public interface for saving file
	public void save()
	{
		saveNet(w_mean_array, hidden_layer);
	}
	
	// Public interface for loading file
	public void load(String file_name)
	{
		try 
		{
			File load_file = new File(file_name);
			loadWeights(load_file);
		}
		catch(Exception e)
		{
			System.out.println(e);
		}
	}

	public void displayNet()
	{
		System.out.println("Weights Mean array");
		for (double[] row : w_mean_array)
		{
			System.out.println(Arrays.toString(row));
		}
		
		System.out.println("Hidden layer");
		for (double[] row : hidden_layer)
		{
			System.out.println(Arrays.toString(row));
		}
		
	}
}

