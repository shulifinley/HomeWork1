package HomeWork1;


import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	public double[] m_coefficients;
	public double m_alpha;

	double theta0;

	//building the LinearRegression object
	LinearRegression(Instances data) throws Exception {
		m_ClassIndex = data.classIndex();
		findAlpha(data);
		m_coefficients = gradientDescent(data);
	}
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		//set m_alpha
		findAlpha(trainingData);
		//find our thetas by running gradientDescent now that we have our alpha
		m_coefficients = gradientDescent(trainingData);
	}


	private void findAlpha(Instances data) throws Exception {
		//according to piazza, uses gradientDescent method below!
		//recitation page 20 for trial alpha values
		m_coefficients = new double [data.numAttributes()];
		double temperror;
		double error = Double.MAX_VALUE;

		for (int i = -17; i <= 1; i++) {
			m_alpha = Math.pow(3,i);
			for (int j = 0; j < 20000; j++) {
				gradientDescent(data);
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
	//refer to lecture 1, slide 33
	//initialize thetas to 0 or 1 (student suggested 0 and ben prefers 1)
	//For all j, update thetaj = thetaj - alpha*partial derivative
	//simultaneous updates using temps - recitation 1 page 14
	//minimizes average square error -> we use calculateMSE method below to calculate the MSE
	private double[] gradientDescent(Instances trainingData) throws Exception {

		double [] tempThetaArr = new double[m_truNumAttributes];
		double sumj = theta0;
		double temperror;
		double error = 0;

		//fill an array of the theta temp values for simultaneous update
		for (int i = 0; i < 5; i++) {
			System.out.println("why arent i printing in gradientDescent? ");

			for (int j = 1; j < trainingData.numInstances(); j++) {
				System.out.println("j =  " + j);

				sumj += (regressionPrediction(trainingData.instance(j)) -
						trainingData.instance(j).value(m_ClassIndex));
				//handling theta0 separately
				if (i != 0) sumj *= trainingData.instance(j).value(i);
				System.out.println("sumj =  " + sumj);

			}
			tempThetaArr[i] = m_coefficients[i] - (m_alpha / trainingData
					.numInstances() * sumj);

			temperror = calculateMSE(trainingData);
			System.out.println("temperror in gradientDescent is: " + temperror);

			if (error - temperror >= 0.003) {
				error = temperror;
				for (int k = 0; k < m_truNumAttributes; k++) {
					m_coefficients[k] = tempThetaArr[k];
				}
			}
			else {
				System.out.println("m_coefficients in gradientDescent is: " + m_coefficients);
				return m_coefficients;
			}
		}
		//copy the thetas that we found that minimize the squared error into m_coefficients global variable
		for (int k = 0; k < m_truNumAttributes; k++) {
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
		for (int i = 0; i < m_coefficients.length - 1; i++) {
			sum += (instance.value(i) * m_coefficients[i + 1]);
		}

		return sum;
	}


	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param data
	 * @return mse
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		//implementation of formula to minimize squared error given in lecture and recitation
		double mse = 0;

		for (int i = 0; i < data.numInstances() ; i++){
			//h(theta(x^(i)) - y^(i)
			mse += Math.pow((regressionPrediction(data.instance(i))
					- data.instance(i).value(m_truNumAttributes)), 2);
		}
		return mse / (2 * data.numInstances());
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