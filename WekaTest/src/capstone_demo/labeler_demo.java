/*
 * Authors: Matthew Quintana and Joshua Schaffer 
 * Date: February 7th, 2018 
 * 
 * The functionality of this program aims to take in a speech sample in order to analyze it 
 * and use the resulting measurements to make an estimate at the English proficiency level 
 * of the sample's speaker. 
 * 
 * General Workflow: 
 * 
 * 1. The audio file is passed into the program, whether through a command line argument, or 
 * through a GUI interface.
 * 
 * 2. The file is then passed into the AuToBI java program where it will be run through an analysis
 * which will produce arff files. 
 * 
 * 3. The attributes and measurements generated from the analysis will be passed into a previously 
 * trained neural net that will pass the measurements through the net and then determine what the output 
 * measurement is. 
 * 
 * 4. This output measurement will be used to classify the audio sample as one of four proficiency levels.
 * In descending order from high proficiency to lower
 * CPE - Certificate of Proficiency in English		C2
 * CAE - Certificate of Advanced English 			C1
 * FCE - First Certificate in English				B2
 * PET - Preliminary English						B1
 * 
 */
package capstone_demo;

import java.lang.*;
import java.io.File;
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

public class labeler_demo {
    
    
    public static void main(String[] args) throws Exception{
        
        // Create the WekaRunner to do the feature selection 
        WekaRunner weka_runner = new WekaRunner();
        
        // Set the training and testing sets of files  
        File training_directory = new File("C:\\Users\\Matt Q\\eclipse-workspace\\AuToBI\\learning_sets\\training_set");
        File testing_directory = new File("C:\\Users\\Matt Q\\eclipse-workspace\\AuToBI\\learning_sets\\testing_set");
        
        /* Training the net example */
        // CAE Proficiencies
        // Female
        //File carole_arff = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/FCAE_0026Carole/phrase_accent_classifier/FCAE_0026Carole_phac_analysis.arff");
        File carole_arff = new File(training_directory.getAbsolutePath() + "/FCAE_0026Carole/phrase_accent_classifier/FCAE_0026Carole_phac_analysis.arff");
		File elisabeth_arff = new File(training_directory.getAbsolutePath() + "/FCAE_0028Elisabeth/phrase_accent_classifier/FCAE_0028Elisabeth_phac_analysis.arff");
		File elodie_arff = new File(training_directory.getAbsolutePath() + "/FCAE_0029Elodie/phrase_accent_classifier/FCAE_0029Elodie_phac_analysis.arff");
		File ryoko_arff = new File(training_directory.getAbsolutePath() + "/FCAE_0029Ryoko/phrase_accent_classifier/FCAE_0029Ryoko_phac_analysis.arff");
		File helena_arff = new File(training_directory.getAbsolutePath() + "/FCAE_0030Helena/phrase_accent_classifier/FCAE_0030Helena_phac_analysis.arff");
		File ebba_arff = new File(training_directory.getAbsolutePath() + "\\FCAE_0031Ebba\\phrase_accent_classifier\\FCAE_0031Ebba_phac_analysis.arff");
		File miriam_arff = new File(training_directory.getAbsolutePath() + "\\FCAE_0033Miriam\\phrase_accent_classifier\\FCAE_0033Miriam_phac_analysis.arff");
		File sangga_arff = new File(training_directory.getAbsolutePath() + "\\FCAE_0033SangGa\\phrase_accent_classifier\\FCAE_0033SangGa_phac_analysis.arff");
		
		// Male
		File manuel_arff = new File(training_directory.getAbsolutePath() + "\\MCAE_0026Manuel\\phrase_accent_classifier\\MCAE_0026Manuel_phac_analysis.arff");
		
		// CPE Proficiency
		File cindy_arff = new File(training_directory.getAbsolutePath() + "\\FCPE_0012Cindy\\phrase_accent_classifier\\FCPE_0012Cindy_phac_analysis.arff");
		File natasha_arff = new File(training_directory.getAbsolutePath() + "\\FCPE_0013Natasha\\phrase_accent_classifier\\FCPE_0013Natasha_phac_analysis.arff");
		File cecilia_arff = new File(training_directory.getAbsolutePath() + "\\FCPE_0014Cecilia\\phrase_accent_classifier\\FCPE_0014Cecilia_phac_analysis.arff");
		
		File manaf_arff = new File(training_directory.getAbsolutePath() + "\\MCPE_0012Manaf\\phrase_accent_classifier\\MCPE_0012Manaf_phac_analysis.arff");
		File rodrigo_arff = new File(training_directory.getAbsolutePath() + "\\MCPE_0013Rodrigo\\phrase_accent_classifier\\MCPE_0013Rodrigo_phac_analysis.arff");
		File rolf_arff = new File(training_directory.getAbsolutePath() + "\\MCPE_0018Rolf\\phrase_accent_classifier\\MCPE_0018Rolf_phac_analysis.arff");
		
		// FCE Proficiency
		File angela_arff = new File(training_directory.getAbsolutePath() + "\\FFCE_0039Angela\\phrase_accent_classifier\\FFCE_0039Angela_phac_analysis.arff");
		File catalania_arff = new File(training_directory.getAbsolutePath() + "\\FFCE_0040Catalania\\phrase_accent_classifier\\FFCE_0040Catalania_phac_analysis.arff");
		File alice_arff = new File(training_directory.getAbsolutePath() + "\\FFCE_0041Alice\\phrase_accent_classifier\\FFCE_0041Alice_phac_analysis.arff");
		
		File alphonso_arff = new File(training_directory.getAbsolutePath() + "\\MFCE_0038Alphonso\\phrase_accent_classifier\\MFCE_0038Alphonso_phac_analysis.arff");
		File charles_arff = new File(training_directory.getAbsolutePath() + "\\MFCE_0038Charles\\phrase_accent_classifier\\MFCE_0038Charles_phac_analysis.arff");
		File hugo_arff = new File(training_directory.getAbsolutePath() + "\\MFCE_0039Hugo\\phrase_accent_classifier\\MFCE_0039Hugo_phac_analysis.arff");
		
		// PET Proficiency
		File annabeatriz_arff = new File(training_directory.getAbsolutePath() + "\\FPET_0027AnnaBeatriz\\phrase_accent_classifier\\FPET_0027AnnaBeatriz_phac_analysis.arff");
		File ekaterina_arff = new File(training_directory.getAbsolutePath() + "\\FPET_0027Ekaterina\\phrase_accent_classifier\\FPET_0027Ekaterina_phac_analysis.arff");
		File carolina_arff = new File(training_directory.getAbsolutePath() + "\\FPET_0030Carolina\\phrase_accent_classifier\\FPET_0030Carolina_phac_analysis.arff");
		
		File andrew_arff = new File(training_directory.getAbsolutePath() + "\\MPET_0026Andrew\\phrase_accent_classifier\\MPET_0026Andrew_phac_analysis.arff");
		File saud_arff = new File(training_directory.getAbsolutePath() + "\\MPET_0026Saud\\phrase_accent_classifier\\MPET_0026Saud_phac_analysis.arff");
		File addul_arff = new File(training_directory.getAbsolutePath() + "\\MPET_0028Addul\\phrase_accent_classifier\\MPET_0028Addul_phac_analysis.arff");
		
		
		
		/* Testing the net */
		// CAE Proficiency Female
		File marilia_arff = new File(testing_directory.getAbsolutePath() + "\\FCAE_0052Marilia\\phrase_accent_classifier\\FCAE_0052Marilia_phac_analysis.arff");
		
		
		// Setting up the set of files to use to determine most common proficiencies
		File[] files = {carole_arff, elisabeth_arff, ryoko_arff, elodie_arff, helena_arff, ebba_arff, miriam_arff, sangga_arff, manuel_arff, 
				cindy_arff, natasha_arff, cecilia_arff, manaf_arff, rodrigo_arff, rolf_arff, angela_arff, catalania_arff, alice_arff, alphonso_arff, charles_arff, 
				hugo_arff, annabeatriz_arff, ekaterina_arff, carolina_arff, andrew_arff, saud_arff, addul_arff};
		
		
		// Determine the most important features over the set
		// Returns their indices in the arff file
		List<Integer[]> attr_count = weka_runner.frequencyCounter(files);
		
		
		// Get the top ten attributes that have appeared the most
		int[] selected_attr = new int[10];
		for (int i = 0; i < 10; i++)
		{
			selected_attr[i] = attr_count.get(i)[0];
		}

		// Display what they were 
		System.out.println(Arrays.toString(selected_attr));
		weka_runner.saveAttributes(selected_attr);
		
		
		// CAE
		double[] carole_mean_array = weka_runner.buildMeanArray(carole_arff, selected_attr);		
		double[] elisabeth_mean_array = weka_runner.buildMeanArray(elisabeth_arff, selected_attr);
		double[] elodie_mean_array = weka_runner.buildMeanArray(elodie_arff, selected_attr);
		double[] ryoko_mean_array = weka_runner.buildMeanArray(ryoko_arff, selected_attr);
		double[] helena_mean_array = weka_runner.buildMeanArray(helena_arff, selected_attr);
		double[] ebba_mean_array = weka_runner.buildMeanArray(ebba_arff, selected_attr);
		double[] miriam_mean_array = weka_runner.buildMeanArray(miriam_arff, selected_attr);
		double[] sangga_mean_array = weka_runner.buildMeanArray(sangga_arff, selected_attr);
		double[] manuel_mean_array = weka_runner.buildMeanArray(manuel_arff, selected_attr);
		
		// CPE
		double[] cindy_mean_array = weka_runner.buildMeanArray(cindy_arff, selected_attr);
		double[] natasha_mean_array = weka_runner.buildMeanArray(natasha_arff, selected_attr);
		double[] cecilia_mean_array = weka_runner.buildMeanArray(cecilia_arff, selected_attr);
		double[] manaf_mean_array = weka_runner.buildMeanArray(manaf_arff, selected_attr);
		double[] rodrigo_mean_array = weka_runner.buildMeanArray(rodrigo_arff, selected_attr);
		double[] rolf_mean_array = weka_runner.buildMeanArray(rolf_arff, selected_attr);
		
		// FCE
		double[] angela_mean_array = weka_runner.buildMeanArray(angela_arff, selected_attr);
		double[] catalania_mean_array = weka_runner.buildMeanArray(catalania_arff, selected_attr);
		double[] alice_mean_array = weka_runner.buildMeanArray(alice_arff, selected_attr);
		double[] alphonso_mean_array = weka_runner.buildMeanArray(alphonso_arff, selected_attr);
		double[] charles_mean_array = weka_runner.buildMeanArray(charles_arff, selected_attr);
		double[] hugo_mean_array = weka_runner.buildMeanArray(hugo_arff, selected_attr);
		
		
		// PET
		double[] annabeatriz_mean_array = weka_runner.buildMeanArray(annabeatriz_arff, selected_attr);
		double[] ekaterina_mean_array = weka_runner.buildMeanArray(ekaterina_arff, selected_attr);
		double[] carolina_mean_array = weka_runner.buildMeanArray(carolina_arff, selected_attr);
		double[] andrew_mean_array = weka_runner.buildMeanArray(andrew_arff, selected_attr);
		double[] saud_mean_array = weka_runner.buildMeanArray(saud_arff, selected_attr);
		double[] addul_mean_array = weka_runner.buildMeanArray(addul_arff, selected_attr);
		
		double[] marilia_mean_array = weka_runner.buildMeanArray(marilia_arff, selected_attr);
		
		
		
		System.out.println("Running net now.");
		NeuralNetwork phac_net = new NeuralNetwork(carole_mean_array);
		
		phac_net.processInput(carole_mean_array, 0);
		phac_net.processInput(elisabeth_mean_array, 0);
		phac_net.processInput(angela_mean_array, 2);
		phac_net.processInput(elodie_mean_array, 0);
		phac_net.processInput(catalania_mean_array, 2);
		phac_net.processInput(ryoko_mean_array, 0);
		phac_net.processInput(natasha_mean_array, 1);
		phac_net.processInput(helena_mean_array, 0);
		phac_net.processInput(andrew_mean_array, 3);
		phac_net.processInput(saud_mean_array, 3);
		phac_net.processInput(ebba_mean_array, 0);
		phac_net.processInput(annabeatriz_mean_array, 3);
		phac_net.processInput(miriam_mean_array, 0);
		phac_net.processInput(rodrigo_mean_array, 1);
		phac_net.processInput(rolf_mean_array, 1);
		phac_net.processInput(sangga_mean_array, 0);
		phac_net.processInput(charles_mean_array, 2);
		phac_net.processInput(carolina_mean_array, 3);
		phac_net.processInput(manuel_mean_array, 0);
		phac_net.processInput(alice_mean_array, 2);
		phac_net.processInput(addul_mean_array, 3);
		phac_net.processInput(hugo_mean_array, 2);
		phac_net.processInput(cecilia_mean_array, 1);
		phac_net.processInput(manaf_mean_array, 1);
		phac_net.processInput(ekaterina_mean_array, 3);
		phac_net.processInput(alphonso_mean_array, 2);
		phac_net.processInput(cindy_mean_array, 1);
		
		
		// NeuralNetwork phac_net = new NeuralNetwork(carole_mean_array);
		phac_net.save();
		
		//phac_net.displayNet();
		phac_net.load("net_weights.dat");
		
		phac_net.displayNet();

		System.out.println("Marilia test ");
		phac_net.testInput(marilia_mean_array, 0);
		
		// Record how many were correctly guessed out of the total
		// Return a percentage
		// Try 
    }  
}
