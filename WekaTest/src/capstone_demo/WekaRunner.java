package capstone_demo;


import java.lang.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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
	

	protected static int[] useLowLevel(Instances data) throws Exception {
	    System.out.println("\n2. Low-level");
	    AttributeSelection attsel = new AttributeSelection();
	    CfsSubsetEval eval = new CfsSubsetEval();
	    GreedyStepwise search = new GreedyStepwise();
	    search.setSearchBackwards(false);
	    attsel.setEvaluator(eval);
	    attsel.setSearch(search);
	    attsel.SelectAttributes(data);
	    int[] indices = attsel.selectedAttributes();
	    System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices) + "\n");
	    return indices;
	}
	
	// Takes in a list of files and 
	public static List<Integer[]> frequencyCounter(File[] analysis_files) throws Exception
	{
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
			// Show what attributes were chosen
			// System.out.println(Utils.arrayToString(weka_attr));
			
			// Add in the attribute index into the array
			// This is where we would do some counting and sorting 
			for (int j = 0; j < weka_attr.length; j++) 
			{
				System.out.println(weka_attr[j]);
				// Check every entry in the already seen attributes
				for (Integer[] entry : best_attr)
				{
					// Check if it's been seen before
					System.out.println("In for loop");
					System.out.println(entry);
					if (entry[0] == weka_attr[j])
					{
						// Increment the count
						entry[1] += 1;
					}
					// If not, add into array. 
					else
					{
						// Make a new entry
						Integer[] new_entry = {weka_attr[j], 1};
						best_attr.add(new_entry);
					}
				}
			}
		}
		
		for (Integer[] row: best_attr)
		{
			System.out.println(row[0].toString());
		}
		
		System.out.println(best_attr.toString());
		
		return best_attr;
	}
	
	public static void main(String[] args) throws Exception
	{
		File carole_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0026Carole/phrase_accent_classifier/FCAE_0026Carole_phac_analysis.arff");
		File elisabeth_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0028Elisabeth/phrase_accent_classifier/FCAE_0028Elisabeth_phac_analysis.arff");
		File ryoko_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0029Ryoko/phrase_accent_classifier/csv_result-FCAE_0029Ryoko_phac_analysis.csv");
		
		File[] files = {carole_arff, elisabeth_arff, ryoko_arff};
		
		frequencyCounter(files);	
	}
}


