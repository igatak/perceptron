package net.tamagothic.machinelearning.perceptron;

import org.apache.commons.math3.linear.BlockRealMatrix;

public class SimplePerceptron {
	private final double learningRate;
	private final BlockRealMatrix weight;

	public SimplePerceptron(int inputUnitSize, int outputUnitSize, double learningRate) {
		this.learningRate = learningRate;
		weight = new BlockRealMatrix(inputUnitSize + 1, outputUnitSize);
		init();
	}

	public void init() {
		for (int i = 0; i < weight.getRowDimension(); i++) {
			for (int j = 0; j < weight.getColumnDimension(); j++) {
				weight.setEntry(i, j, Math.random());
			}
		}
	}

	public double train(double[] input, double[] goldStandard) {
		// 入力値から拡張特徴ベクトルに変換
		BlockRealMatrix augmentedInputLayer = createAugmentedFeatureVector(input);

		BlockRealMatrix outputLayer = forwardPropagete(augmentedInputLayer);

		updateWeight(augmentedInputLayer.getRow(0), outputLayer.getRow(0), goldStandard);

		return computeError(outputLayer.getRow(0), goldStandard);
	}

	private BlockRealMatrix forwardPropagete(BlockRealMatrix augmentedInputLayer) {
		// 入力層×重み行列
		BlockRealMatrix outputLayer = augmentedInputLayer.multiply(weight);
		return outputLayer;
	}

	private BlockRealMatrix createAugmentedFeatureVector(double[] input) {
		double[] augmentedVector = new double[input.length + 1];
		augmentedVector[0] = 1D;
		System.arraycopy(input, 0, augmentedVector, 1, input.length);
		BlockRealMatrix augmentedInputLayer = new BlockRealMatrix(1, augmentedVector.length);
		augmentedInputLayer.setRow(0, augmentedVector);
		return augmentedInputLayer;
	}

	private void updateWeight(double[] augmentedInput, double[] output, double[] goldStandard) {
		for (int i = 0; i < weight.getRowDimension(); i++) {
			for (int j = 0; j < weight.getColumnDimension(); j++) {
				double delta = learningRate * (goldStandard[j] - output[j]) * augmentedInput[i];
				weight.addToEntry(i, j, delta);
			}
		}
	}

	public double[] classify(double[] input) {
		BlockRealMatrix augmentedInputLayer = createAugmentedFeatureVector(input);
		BlockRealMatrix output = forwardPropagete(augmentedInputLayer);
		double[] result = new double[output.getColumnDimension()];
		for (int i = 0; i < result.length; i++) {
			result[i] = output.getEntry(0, i) > 0.5 ? 1 : 0;
		}
		return result;
	}

	private double computeError(double[] output, double[] goldStandard) {
		double error = 0;
		for (int i = 0; i < output.length; i++) {
			error += Math.pow(goldStandard[i] - output[i], 2);
		}
		return error / output.length;
	}
}
