package net.tamagothic.machinelearning.perceptron;

public class OrOperationPerceptron {
	private final SimplePerceptron perceptron;

	public OrOperationPerceptron(int inputUnitSize, int outputUnitSize, double learningRate) {
		this.perceptron = new SimplePerceptron(inputUnitSize, outputUnitSize, learningRate);
	}

	public void train() {
		for (int i = 0; i < 100; i++) {
			double error = 0;
			error += perceptron.train(new double[] {
					0, 0
			}, new double[] {
				0
			});
			error += perceptron.train(new double[] {
					0, 1
			}, new double[] {
				1
			});
			error += perceptron.train(new double[] {
					1, 0
			}, new double[] {
				1
			});
			error += perceptron.train(new double[] {
					1, 1
			}, new double[] {
				1
			});
			error /= 4;
			//System.out.println(error);
		}
	}

	public double operate(double[] input) {
		return perceptron.classify(input)[0];
	}

	public static void main(String[] args) {
		OrOperationPerceptron operator = new OrOperationPerceptron(2, 1, 0.3);
		operator.train();

		System.out.println(operator.operate(new double[] {
				0, 0
		}));
		System.out.println(operator.operate(new double[] {
				0, 1
		}));
		System.out.println(operator.operate(new double[] {
				1, 0
		}));
		System.out.println(operator.operate(new double[] {
				1, 1
		}));
	}

}
