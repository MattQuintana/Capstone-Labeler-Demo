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
    
    private static double[] buildMeanArray(String path) throws Exception {
    	ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
    	Instances data = source.getDataSet();

        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
        
        int num_attributes = data.numAttributes();
        
        // low-level
        int[] selected_att_indices;
        selected_att_indices = useLowLevel(data);

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
    	return att_mean_values;
	}   	

    
    public static void main(String[] args) throws Exception{
        String user = System.getProperty("user.name");
        System.out.println(user);
        File autobi_directory = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI");
        
        // Build command to run AuToBI jar file on an audio file. 
        String[] command_list = {"java", "-jar", "AuToBI.jar", "-wav_file=./cambridge-test/FCAE_0026Carole.wav", "-boundary_tone_classifier=bdc_burnc.pabt.classification.model", "-out_file=./analysis_results/FCAE_0026Carole/boundary_tone_classifier/java_out.txt"};
        
        ProcessBuilder pb = new ProcessBuilder(command_list);
        pb.directory(autobi_directory);
        pb.inheritIO();
        pb.command(command_list);
        
        // Print out the command structure and parameters
        //System.out.println("" + pb.command());
        
        Process p = pb.start();
        
        // load data
        System.out.println("\n0. Loading data");
        
        double[] carole_arff = buildMeanArray("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0026Carole/phrase_accent_classifier/FCAE_0026Carole_phac_analysis.arff");
        double[] elisabeth_arff = buildMeanArray("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0028Elisabeth/phrase_accent_classifier/FCAE_0028Elisabeth_phac_analysis.arff");
        
        NeuralNetwork analysis_net = new NeuralNetwork(carole_arff);
        
        analysis_net.processInput(carole_arff, 0);
        
        NeuralNetwork elis_net = new NeuralNetwork(elisabeth_arff);
        elis_net.processInput(elisabeth_arff, 0);
    }  
}
