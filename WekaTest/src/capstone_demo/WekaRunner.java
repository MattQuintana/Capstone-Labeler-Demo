package capstone_demo;


import java.lang.*;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.experiment.Stats;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.converters.CSVSaver;
import weka.core.converters.CSVLoader;

public class WekaRunner {
	
	
	// Runs the Weka attribute selection and returns a set of indices of important attributes 
	protected static int[] useLowLevel(Instances data) throws Exception {
	    //System.out.println("\n2. Low-level");
	    AttributeSelection attsel = new AttributeSelection();
	    CfsSubsetEval eval = new CfsSubsetEval();
	    GreedyStepwise search = new GreedyStepwise();
	    search.setSearchBackwards(false);
	    attsel.setEvaluator(eval);
	    attsel.setSearch(search);
	    attsel.SelectAttributes(data);
	    int[] indices = attsel.selectedAttributes();
	    //System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices) + "\n");
	    return indices;
	}
	
	
	
	// Takes in a list of files, performs a feature analysis, and counts the 
	// frequency by which attributes are chosen.
	// Returns a list that contains the different attribute indices and their frequency count 
	public static List<Integer[]> frequencyCounter(File[] analysis_files) throws Exception
	{
		// Create a list to hold the attribute indices and it's count  
		List<Integer[]> best_attr = new ArrayList<Integer[]>();
		
		// Run the analysis on several files 
		for (int i = 0; i < analysis_files.length; i++)
		{
			
			// Get the current file to analyze
			// Convert it to a instances class
			File current_file = analysis_files[i];
			ConverterUtils.DataSource source = new ConverterUtils.DataSource(current_file.getPath());
			Instances file_data = source.getDataSet();
			
			// Pass the data into the analysis
			int[] weka_attr = useLowLevel(file_data);
			
			// Add in the attribute index into the array
			// This is where we would do some counting and sorting 
			for (int j = 0; j < weka_attr.length; j++) 
			{
				
				// Boolean to check if it's seen an attribute before
				Boolean found_previous = false; 
				
				// Check every entry in the already seen attributes
				for (Integer[] entry : best_attr)
				{
					// Check if it's been seen before
					if (entry[0] == weka_attr[j])
					{
						// Increment the count
						entry[1] += 1;
						found_previous = true;
						break;
					}
				}
				// If it searched through the entire list and didn't find the entry
				if (found_previous == false)
				{
					// Make a new entry
					Integer[] new_entry = {weka_attr[j], 1};
					best_attr.add(new_entry);
				}								
			}
		}
		
		// Build a comparator to sort the list by count
		Comparator<Integer[]> sorter = new Comparator<Integer[]>()
		{
			@Override
			public int compare(Integer[] o1, Integer[] o2) {
	            int res = o2[1].compareTo(o1[1]);
	            return res;
	        }
		};
		
		// Sort the list by count, descending order
		best_attr.sort(sorter);
		
		// Print out each attribute and its count. 
		/*
		for (Integer[] row: best_attr)
		{
			System.out.println(row[0].toString() + " " + row[1].toString());
		}*/
		return best_attr;
	}
	
	// Create an array of mean values for each chosen attribute index
	public static double[] buildMeanArray(File arff_file, int[] indices) throws Exception
	{
		double[] mean_value_array = new double[indices.length];
		
		// Get a usable data
		ConverterUtils.DataSource source = new ConverterUtils.DataSource(arff_file.getPath());
    	Instances data = source.getDataSet();
    	
    	// For every index 
    	for(int i = 0; i < indices.length; i++) {
            //get an AttributeStats object
            AttributeStats attStats = data.attributeStats(i);
            Attribute selected_att = data.attribute(indices[i]);
            // Check that we can find the numeric value of the attribute
        	if(selected_att.isNumeric()) {
        		Stats s = attStats.numericStats;
        		// Get the mean value
        		mean_value_array[i] = s.mean;
        	}	
        }
    	// Return the set of mean values 
    	return mean_value_array;
	}
	
	// Save the attributes chosen to a file
	public void saveAttributes(int[] selected_attributes)
	{
		File attrs_file = new File("selected_attributes.dat");
		try 
		{
			DataOutputStream outstream = new DataOutputStream(new FileOutputStream(attrs_file));
			for(Integer attribute_index: selected_attributes)
			{
				outstream.writeInt(attribute_index);
			}
		}
		catch (Exception e)
		{
			System.out.println(e);
		}	
	}
	
	
	// Load the attributes from some file previously chosen
	public List<Integer> loadAttributes(File weights_file)
	{
		
		List<Integer> attributes_list = new ArrayList<Integer>();
		try 
		{	
			Scanner scan = new Scanner(weights_file);
			while (scan.hasNextInt())
			{
				attributes_list.add(scan.nextInt());
			}
			
		}
		catch (Exception e)
		{
			//System.out.println(e);
		}	
		
		return attributes_list;
	}
	
	
	public static void main(String[] args) throws Exception
	{
		File carole_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0026Carole/phrase_accent_classifier/FCAE_0026Carole_phac_analysis.arff");
		File elisabeth_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0028Elisabeth/phrase_accent_classifier/FCAE_0028Elisabeth_phac_analysis.arff");
		File elodie_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0029Elodie/phrase_accent_classifier/FCAE_0029Elodie_phac_analysis.arff");
		File ryoko_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0029Ryoko/phrase_accent_classifier/csv_result-FCAE_0029Ryoko_phac_analysis.csv");
		File helena_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0030Helena/phrase_accent_classifier/FCAE_0030Helena_phac_analysis.arff");
		File ebba_arff = new File("C:\\Users\\Matt Q\\eclipse-workspace\\AuToBI\\analysis_results\\FCAE_0031Ebba\\phrase_accent_classifier\\FCAE_0031Ebba_phac_analysis.arff");
		
		File[] files = {carole_arff, elisabeth_arff, ryoko_arff, elodie_arff, helena_arff, ebba_arff};
		
		List<Integer[]> attr_count = frequencyCounter(files);
		
		/*
		for (int i = 0; i < 10; i++)
		{
			System.out.println(attr_count.get(i)[0].toString() + " " + attr_count.get(i)[1].toString());
		}
		*/
		
		int[] selected_attr = new int[10];
		for (int i = 0; i < 10; i++)
		{
			selected_attr[i] = attr_count.get(i)[0];
		}
		
		System.out.println(Arrays.toString(selected_attr));
		
		double[] mean_array = buildMeanArray(carole_arff, selected_attr);
		
		System.out.println(Arrays.toString(mean_array));
	}
}


