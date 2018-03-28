package Homework1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	double theta0;

	//building the LinearRegression object
	LinearRegression(Instances data) throws Exception {
		m_ClassIndex = data.classIndex();
		findAlpha(data);
		m_coefficients = gradientDescent(data);
	}

	// helper method to calculate inner product of theta vector and x vector
	public double innerProduct(Instance instance, double[] thetas) throws Exception {
		double inrPrd = 0;

		for (int j = 1; j < m_truNumAttributes + 1; j++) {
			inrPrd += instance.value(j) * thetas[j - 1];
		}
		return inrPrd;
	}

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		//set m_alpha
		findAlpha(trainingData);
		//find our thetas by running gradientDescent now that we have our alpha
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void findAlpha(Instances data) throws Exception {
		//according to piazza, uses gradientDescent method below!
		//recitation page 20 for trial alpha values
		double temperror = calculateMSE(data);
		double error = temperror;
		double tempThetas[] = new double[m_truNumAttributes + 1];
		for (int i = -17; i < 1; i++) {
			m_alpha = Math.pow(3,i);
			for (int j = 0; j < 20000; j++) {
				tempThetas = gradientDescent(data);
				if (j % 100 == 0) {
					temperror = calculateMSE(data);
					if (temperror < error) {
						error = temperror;
					} else {
						return;
					}
				}
			}
		}
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		double [] tempThetaArr = new double[m_truNumAttributes + 1];
		double sumj = theta0;
		double temperror;
		double error = 0;

		//fill an array of the theta temp values for simultaneous update
		for (int i = 0; i < m_truNumAttributes + 1; i++) {

			for (int j = 0; j < trainingData.size() + 1; j++) {
				sumj += regressionPrediction(trainingData.instance(j)) - trainingData.instance(j).value(m_truNumAttributes);
				//handling theta0 separately
				if (i != 0) sumj *= trainingData.instance(j).value(i);
			}
			tempThetaArr[i] = m_coefficients[i] - m_alpha / trainingData.size() * sumj;

			temperror = calculateMSE(trainingData);

			if (error - temperror >= 0.003) {
				error = temperror;
				for (int k = 0; k < m_truNumAttributes + 1; k++) {
					m_coefficients[k] = tempThetaArr[k];
				}
			}
			else {
				return m_coefficients;
			}
		}
		//copy the thetas that we found that minimize the squared error into m_coefficients global variable
		for (int k = 0; k < m_truNumAttributes + 1; k++) {
			theta0 = tempThetaArr[0];
			m_coefficients[k] = tempThetaArr[k];
		}
		return m_coefficients;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double sum = theta0;
		sum += innerProduct(instance, m_coefficients);
		return sum;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {

		//implementation of formula to minimize squared error given in lecture and recitation
		double mseSum = 0;
		double mse;

		for (int i = 1; i <= data.size(); i++){
			//h(theta(x^(i)) - y^(i)
			mseSum += Math.pow(regressionPrediction(data.instance(i)) -  data.instance(i).value(m_truNumAttributes), 2);
		}
		mse = mseSum / (2 * data.size());
		return mse;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
