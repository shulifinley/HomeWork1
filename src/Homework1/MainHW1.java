package Homework1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		double trainError;
		double testError;
		Instances data;
		for (int t = 0; t < 1; t++) {
			//load data
			data = loadData("wind_training");
			if (t == 1) {
				data = loadData("wind_testing");
			}

			//find best alpha and build classifier with all attributes
			LinearRegression classifier = new LinearRegression(data);

			//build classifiers with all 3 attributes combinations

			double sum = 0;
			double tempError;
			double minError = 0;
			// initialized to 0 arbitrarily, the value will be changed in the first iteration of the k loop
			int[] threeAttributes = new int[3];
			Remove remove = new Remove();
			String[] options = new String[4];
			options [0] = "-V";

			for (int i = 0; i < 13; i++) {
				options[1] = Integer.toString(i);

				for (int j = i + 1; j < 14; j++) {
					options[2] = Integer.toString(j);

					for (int k = j + 1; k < 15; k++) {
						options[3] = Integer.toString(k);

						remove.setOptions(options);
						remove.setInputFormat(data);
						classifier.buildClassifier(Filter.useFilter(data, remove));

						tempError = classifier.calculateMSE(data);

						if (k == 2) minError = tempError;
						if (tempError < minError) {
							minError = tempError;
							threeAttributes[0] = i;
							threeAttributes[1] = j;
							threeAttributes[2] = k;
						}
					}
				}
			}

			if (t==0) trainError = minError;
			if (t==1) testError = minError;
		}
	}
}


