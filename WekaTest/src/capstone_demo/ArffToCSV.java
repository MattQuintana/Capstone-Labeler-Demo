package capstone_demo;

import weka.core.Instances;
import weka.core.converters.*;
import java.io.File;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;

public class ArffToCSV {
	
	public static void main(String[] args) throws Exception
	{
		
		
		
        String test_path = "C:/Users/Matt Q/eclipse-workspace/AuToBI/test_dir/FCAE_0031Ebba_phac_analysis.arff";
        String output_path = "C:/Users/Matt Q/eclipse-workspace/AuToBI/test_dir/FCAE_0031Ebba_phac_analysis_csv.csv";
        
        BufferedReader reader = new BufferedReader(new FileReader(new File(test_path)));
        //ConverterUtils.DataSource source = new ConverterUtils.DataSource(test_path);
        
		//ArffLoader loader = new ArffLoader();
		//loader.setSource(new File(test_path));
		
		Instances data = new Instances(reader);

		//System.out.println(loader.getDataSet());
		
		/*
		Instances data = loader.getDataSet();
		

		
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);
		saver.setFile(new File("C:/Users/Matt Q/eclipse-workspace/AuToBI/test_dir/FCAE_0026Carole_phac_analysis_csv.csv"));
		saver.writeBatch();
		*/
		
		System.out.println("Output file.");
	}
	
	public File convertArfftoCSV(String input_path, String output_path) throws Exception
	{
		Loader loader = new ArffLoader();
		loader.setSource(new File(input_path));
		Instances data = loader.getDataSet();
		
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);
		saver.setFile(new File(output_path));
		
		File csv_file = new File(output_path);
		return csv_file;
	}

}
