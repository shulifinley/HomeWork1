package HomeWork1;


import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SystemInfo;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	public double[] m_coefficients;
	public double m_alpha;


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
		m_coefficients = new double[data.numAttributes()];
		double temperror = Double.MAX_VALUE;
		double error = Double.MAX_VALUE;
	//	boolean stop = false;

				for (int i = -17; i < 1; i++) {
						m_alpha = Math.pow(3, i);
						for (int j = 0; j < 20000; j++) {
					//		if (stop == false) {

								//	System.out.println("find alpha, round number: " + j + " i" +
								//			" = " + i);
								gradientDescent(data);
								//	System.out.println("m coeff " + m_coefficients[5]);


								if ((j % 100) == 0) {
									temperror = calculateMSE(data);

									for (int b = 0; b < m_coefficients.length;
										 b++) {
										System.out.println("coef " + b + " :" +
												" " +
												this.m_coefficients[b]);
									}


									if (temperror > error) {
										System.out.println("??? " );
										//			stop = true;
										System.out.println("temperror = " + temperror);
										System.out.println("error = " + error);
										break;

									} else if (temperror < error) {
										error = temperror;
										System.out.println("error = temperror");
									}
								}
							}

							return;
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
		double[] tempThetaArr = new double[m_coefficients.length];
		double sumj = m_coefficients[0];
		double temperror;
		double error = 0;
		double ip;


		//fill an array of the theta temp values for simultaneous update

		for (int i = 1; i < m_coefficients.length; i++) {
			//initialize the m_coefficients array
			for (int t = 0; t < m_coefficients.length; t++) {
				m_coefficients[t] = 1;
			}
			for (int j = 0; j < trainingData.numInstances(); j++) {

				sumj += (regressionPrediction(trainingData.instance(j)) -
						trainingData.instance(j).value(m_ClassIndex));

				ip = (sumj * trainingData.instance(j).value(i));
				tempThetaArr[i] = m_coefficients[i] -
						(m_alpha / trainingData.numInstances() * ip);
		//		System.out.println("First update (temp theta arr) = " +
		//				tempThetaArr[i]);
			}
		}
				temperror = calculateMSE(trainingData);
//					System.out.println("temp error = " + temperror);
//					System.out.println("error = " + error);

				if (Math.abs(error - temperror) > 0.003) {
					error = temperror;

					for (int k = 0; k < m_coefficients.length ; k++) {
		//				System.out.println("k= " + k);

						m_coefficients[k] = tempThetaArr[k];
		//				System.out.println("1 m_coef = " + m_coefficients[k]);
		//				System.out.println("1 temp theta array = " +
		//						tempThetaArr[k]);

					}
					} else {
						System.out.println("2. m_coef = " + m_coefficients[8]);

							return m_coefficients;
					}



//
//		//copy the thetas that we found that minimize the squared error into m_coefficients global variable
//		for (int k = 0; k < m_truNumAttributes; k++) {
//			m_coefficients[k] = tempThetaArr[k];
//			System.out.println("2 m_coef = " + m_coefficients);
//		}


//			System.out.println("3. m_coef = " + m_coefficients[4]);
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

		double sum = m_coefficients[0];
		for (int i = 1; i < m_coefficients.length; i++) {
			sum += (instance.value(i - 1) * m_coefficients[i]);
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
					- data.instance(i).value(m_ClassIndex)), 2);
		}
		return mse / (2.0 * data.numInstances());
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