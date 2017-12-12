/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package capstone_demo;

import java.lang.*;
import java.io.File;
import java.util.Arrays;
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

/**
 *
 * @author Matt Q
 */
public class labeler_demo {
    
    protected static void useClassifier(Instances data) throws Exception {
        System.out.println("\n1. Meta-classfier");
        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        J48 base = new J48();
        classifier.setClassifier(base);
        classifier.setEvaluator(eval);
        classifier.setSearch(search);
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(evaluation.toSummaryString());
    }
    
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
    
    private static double nonlin(double x) {
    	return 1/(1 + Math.exp(-x));
    }
    
    public static void main(String[] args) throws Exception{
        String user = System.getProperty("user.name");
        System.out.println(user);
        File autobi_directory = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI");
        
        // Build command to run AuToBI jar file on an audio file. 
        String[] command_list = {"java", "-jar", "AuToBI.jar", "-wav_file=FCAE_0026Carole.wav", "-boundary_tone_classifier=bdc_burnc.pabt.classification.model", "-out_file=./analysis_results/java_out.txt"};
        
        ProcessBuilder pb = new ProcessBuilder(command_list);
        pb.directory(autobi_directory);
        pb.inheritIO();
        pb.command(command_list);
        
        // Print out the command structure and parameters
        //System.out.println("" + pb.command());
        
        Process p = pb.start();
        
        // load data
        System.out.println("\n0. Loading data");
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/Phrase Accent Classifier/FC_Carol_PHAC_analysis.arff");
        //ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/Pitch Accent Detector/FC_Carol_PAD_analysis.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
        
        int num_attributes = data.numAttributes();
        
        //Example code of going through the attributes
        /*
        for (int i = 0; i < num_attributes; i++) {
            //check if current attr is of type nominal
            if (data.attribute(i).isNominal()) {
                System.out.println("The "+i+"th Attribute is Nominal"); 
                //get number of values
                int n = data.attribute(i).numValues();
                System.out.println("The "+i+"th Attribute has: "+n+" values");
            }           

            //get an AttributeStats object
            AttributeStats as = data.attributeStats(i);
            int dC = as.distinctCount;
            System.out.println("The "+i+"th Attribute has: "+dC+" distinct values");

            //get a Stats object from the AttributeStats
            if (data.attribute(i).isNumeric()){
                System.out.println("The "+i+"th Attribute is Numeric"); 
                Stats s = as.numericStats;
                System.out.println("The "+i+"th Attribute has min value: "+s.min+" and max value: "+s.max+" and mean value: "+s.mean+" and stdDev value: "+s.stdDev );
            }

        }*/

        // 1. meta-classifier
        //useClassifier(data);
        
        // 2. low-level
        int[] selected_att_indices = useLowLevel(data);
        double[] att_mean_values = new double[selected_att_indices.length];

        for(int i = 0; i < selected_att_indices.length; i++) {
            //get an AttributeStats object
            AttributeStats attStats = data.attributeStats(i);
            Attribute selected_att = data.attribute(selected_att_indices[i]);
        	if(selected_att.isNumeric()) {
        		Stats s = attStats.numericStats;
        		//System.out.println("Attribute: " + selected_att_indices[i] + " has mean: " + s.mean);
        		//System.out.println("Attribute: " + selected_att_indices[i] + " has min: " + s.min);
        		//System.out.println("Attribute: " + selected_att_indices[i] + " has max: " + s.max);
        		//System.out.println("Attribute: " + selected_att_indices[i] + " has stddev: " + s.stdDev);
        		att_mean_values[i] = s.mean;
        	}	
        }
        
        // Beginnings of a neural net
        double[][] w_mean_array = new double[selected_att_indices.length][4];
        
        // Fill matrix with random weights between 0 and 1
        for(int i = 0; i < w_mean_array.length; i++) {
        	for(int j = 0; j < w_mean_array[i].length; j++) {
        		w_mean_array[i][j] = Math.random();
        	}
        }
        
        /////////////////////////////////////////////////////////
        //Here is where the loop of Machine Learning should begin
        int actual_score = 0;
        int guess_score = -1;
        
        // While the program still hasn't made a correct guess on the proficiency level
        while (actual_score != guess_score) {
        	// Output layer to hold final proficiency guesses
        	double[] output_layer = new double[4];
            
            // Do the matrix multiplication
        	// Basically multiply all of the mean values retrieved from the attributes 
        	// by the randomized weights in the matrix. 
            for(int matx_col = 0; matx_col < w_mean_array[0].length; matx_col++) {
            	double col_sum = 0;
            	for(int matx_row = 0; matx_row < w_mean_array.length; matx_row++) {
            		col_sum += att_mean_values[matx_row] * w_mean_array[matx_row][matx_col];
            	}
            	// Put all of those multiplications into the corresponding output layer node
            	output_layer[matx_col] = col_sum;
            }
            
            //System.out.println(Arrays.toString(output_layer));
            
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
            guess_score = max_index;
            
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
            if (guess_score != actual_score) {
            	// Adjust the weights
            	// Lower them by some kind of factor based on the difference between the actual and guess score. 
            	// Lower the weights associated with the wrong guess 
            	for (int matx_row = 0; matx_row < w_mean_array.length; matx_row++) {
            		for (int matx_col = 0; matx_col < w_mean_array[matx_row].length; matx_col++) {
            			if (matx_col == max_index) {
            				w_mean_array[matx_row][max_index] *= 1 - nonlin(Math.abs(actual_score-guess_score));
            				w_mean_array[matx_row][max_index] = nonlin(w_mean_array[matx_row][matx_col]);
            			}
            			else {
            				// Boost the weights that are not associated with the wrong guess
            				w_mean_array[matx_row][matx_col] *= 1 + nonlin(Math.abs(actual_score-guess_score));
            				w_mean_array[matx_row][matx_col] = nonlin(w_mean_array[matx_row][matx_col]);
            			}
            		}
            	}
            }
        }
        
        // Here we would make the guess and then compare against the actual proficiency level/score
        ///////////////////////////////////////////////////////////////////////////////////////////

    }  
    
    
}
