/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package capstone_demo;

import java.lang.*;
import java.io.File;
import java.util.Random;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

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
    
    protected static void useLowLevel(Instances data) throws Exception {
        System.out.println("\n2. Low-level");
        AttributeSelection attsel = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(false);
        attsel.setEvaluator(eval);
        attsel.setSearch(search);
        attsel.SelectAttributes(data);
        int[] indices = attsel.selectedAttributes();
        System.out.println("selected attribute indices (starting with 0):\n" + Utils.arrayToString(indices));
    }
    
    public static void main(String[] args) throws Exception{
        String user = System.getProperty("user.name");
        System.out.println(user);
        //String cmd = "java -jar C:/Users/Matt Q/eclipse-workspace/AuToBI/AuToBI.jar -wav_file=C:/Users/Matt Q/eclipse-workspace/AuToBI/FCAE_0026Carole.wav -boundary_tone_classifier=bdc_burnc.pabt.classification.model -out_file=C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/hello.txt";
        //Runtime.getRuntime().exec(cmd);
        File autobi_directory = new File("C:/Users/Matt Q/eclipse-workspace/AuToBI");
        
        String[] command_list = {"java", "-jar", "AuToBI.jar", "-wav_file=FCAE_0026Carole.wav", "-boundary_tone_classifier=bdc_burnc.pabt.classification.model", "-out_file=./analysis_results/java_out.txt"};
        
        ProcessBuilder pb = new ProcessBuilder(command_list);
        pb.directory(autobi_directory);
        pb.inheritIO();
        pb.command(command_list);
        
        System.out.println("" + pb.command());
        
        Process p = pb.start();
        
        // load data
        System.out.println("\n0. Loading data");
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:/Users/Matt Q/eclipse-workspace/AuToBI/analysis_results/Phrase Accent Classifier/FC_Carol_PHAC_analysis.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);

        // 1. meta-classifier
        //useClassifier(data);
        
        // 2. low-level
        useLowLevel(data);
    }  
    
    
}
