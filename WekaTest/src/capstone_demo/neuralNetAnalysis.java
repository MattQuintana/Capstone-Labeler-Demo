package capstone_demo;

import java.io.*;
import java.util.List;



public class neuralNetAnalysis {
	
	public static void main(String[] args) throws Exception{

		System.out.println(args.toString());
		// Create the WekaRunner for running the analysis 
        WekaRunner weka_runner = new WekaRunner();
        File attrs_file = new File("selected_attributes.dat");
        List<Integer> selected_attributes = weka_runner.loadAttributes(attrs_file);
        int selected_indices[] = new int[selected_attributes.size()];
        
        
        for (int i = 0; i < selected_attributes.size(); i++)
        {
        	selected_indices[i] = selected_attributes.get(i);
        }
        
		int proficiency_level = 0;
		
		// Get an arff file from the command line
		File arff_input = new File(args[0]);
	
		// Parse the file input to figure out which type of proficiency it is
		if (args[0].startsWith("CAE",1))
		{
			proficiency_level = 0;
		}
		else if (args[0].startsWith("CPE", 1))
		{
			proficiency_level = 1;
		}
		else if (args[0].startsWith("FCE", 1))
		{
			proficiency_level = 2;
		}
		else if (args[0].startsWith("PET", 1))
		{
			proficiency_level = 3;
		}
		 
		// Change it to a mean array
		double[] arff_mean_array = weka_runner.buildMeanArray(arff_input, selected_indices);
        // Create the neural net to process the input files
        NeuralNetwork net = new NeuralNetwork(arff_mean_array);
        net.load("net_weights.dat");
		
		// Test it in the net with its corresponding proficiency level
        
        net.testInput(arff_mean_array, proficiency_level);

	}

}
