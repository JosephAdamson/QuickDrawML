package neuralnetwork;

import utils.Matrix;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * Feedforward neural network model used to classify a subset of images
 * from the Google Quick, Draw! dataset.
 * 
 * @author Joseph Adamson
 */
public class NeuralNetwork {
    
    /**
     * The number of nodes in the first layer; the input data.
     */
    private final int inputNodes;

    /**
     * The number of nodes in the final layer; the final outputs.
     */
    private final int outputNodes;

    /**
     * The number of network layers.
     */
    private final int layerNumber;

    /**
     * The number of Layers a network requires; includes all hidden
     * and output layers. The first input 'layer' is omitted as it 
     * is merely data (no weights or bias).
     */
    private final Layer[] layers;

    /**
     * A matrix array; each z matrix is the dot product of the weights 
     * feeding in the the current layer and the previous activations plus
     * the bias E.g. w1a1 + w2a2 ... + wnan + b
     */
    private final Matrix[] zl;

    /**
     * A matrix array; to store the activations σ(zl).
     */
    private final Matrix[] activations;

    /**
     * Our default activation function and its derivative.
     */
    public static final Function<Double, Double> SIGMOID = (x) -> 1 / (1 + Math.exp(-x));
    public static final Function<Double, Double> SIGMOIDPRIME =
            (x) -> Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)));
    
    /**
     * Constructs a neural network; the number of layers (excluding input)
     * are defined by the size of the layerSizes array.
     *
     * @param inputs the number of inputs (the first layer) into the network.
     * @param layerSizes an array (vargs) of integers; each index corresponds 
     * to layer (zero indexed) where each element is the amount
     * of output nodes in that particular layer.
     */
    public NeuralNetwork(int inputs, int... layerSizes) {
        for (int layerSize : layerSizes) {
            if (layerSize < 1) {
                throw new IllegalArgumentException("There must be " +
                        "at least one node in each layer");
            }
        }
        this.inputNodes = inputs;
        this.outputNodes = layerSizes[layerSizes.length - 1];
        this.layerNumber = layerSizes.length + 1;
        this.layers = new Layer[layerSizes.length];

        layers[0] = new Layer(inputNodes, layerSizes[0]);
        for (int i = 1; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i - 1], layerSizes[i]);
        }

        // z and activation matrices initialized at zero.
        this.zl = new Matrix[layerSizes.length];
        this.activations = new Matrix[layerSizes.length + 1];
    }

    /**
     * Clone a network using existing layers
     * 
     * @param layerConfig: Pre-trained parameters.
     */
    public NeuralNetwork(Layer[] layerConfig) {
        this.inputNodes = layerConfig[0].weights.getCols();
        this.outputNodes = layerConfig[layerConfig.length -1].weights.getRows();
        this.layers = layerConfig;
        this.layerNumber = layerConfig.length + 1;

        this.zl = new Matrix[layerConfig.length];
        this.activations = new Matrix[layerConfig.length + 1];
    }

    /**
     * Save the parameters of a trained network to a serializable 
     * file
     */
    public void saveNetwork() {
        try {
            FileOutputStream fos = new FileOutputStream("networkModel.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            Layer[] model = this.layers;
            oos.writeObject(model);
        } catch (Exception IOException) {
            System.err.println("Directory no found");
        } 
    }

    /**
     * Create Network with pre-existing parameters.
     * @param network parameter .dat file
     * @return a NeuralNetwork
     */
    public static NeuralNetwork loadNetwork(File network) throws IOException {
        Layer[] output = null;
        try {
            FileInputStream fis = new FileInputStream(network);
            ObjectInputStream ois = new ObjectInputStream(fis);
            output = (Layer[]) ois.readObject();
            
            
        } catch (Exception e) {
            System.err.println("File not found");
        }
        
        if (output == null) {
            throw new IOException("File not found");
        } else {
            return new NeuralNetwork(output);
        }
    }

    /**
     * @return number of input nodes for the network.
     */
    public int getInputNodes() {
        return inputNodes;
    }

    /**
     * @return number of output nodes for the network.
     */
    public int getOutputNodes() {
        return outputNodes;
    }
    
    public int getLayerNumber() {
        return layerNumber;
    }

    /**
     * Trains network using mini-batch gradient descent.
     * 
     * @param trainingData preprocessed data; an arraylist of subarray 'annotations',
     * @param epochs The number of passes through the full training data.
     * @param batchSize the number of sub-batches for the training data to be spit into
     * @param alpha the training rate.
     * @param lambda the regularization constant.
     */
    public double[][] mbgd(ArrayList<ArrayList<Matrix>> trainingData, int epochs,
                           int batchSize, double alpha, double lambda, ArrayList<ArrayList<Matrix>> validationData) {

        double[][] performance = new double[4][epochs];

        for (int i = 0; i < epochs; i++) {
            Collections.shuffle(trainingData);

            int batches = trainingData.size() / batchSize;
            for (int j = 0; j < batches; j++) {
                int start = j * batchSize;
                List<ArrayList<Matrix>> batch =
                        trainingData.subList(start, start + batchSize);
                updateWithBatch(batch, alpha, lambda, trainingData.size());
            }

            // Log accuracies and cost for each epoch
            double validationCost = evaluateCost(validationData, lambda);
            double validationAcc = evaluateAccuracy(validationData);
            double trainingCost = evaluateCost(trainingData, lambda);
            double trainingAcc = evaluateAccuracy(trainingData);

            performance[0][i] = trainingCost;
            performance[1][i] = validationCost;
            performance[2][i] = trainingAcc;
            performance[3][i] = validationAcc;

            System.out.printf("Epoch %d/%d%n%d/%d [==================]" +
                            " training: - cost: %.5f - acc: %.5f, validation: - cost: %.5f - acc: %.5f%n",
                    i, epochs, trainingData.size(), trainingData.size(),
                    trainingCost, trainingAcc, validationCost, validationAcc);
        }
        return performance;

    }

    /**
     * Update the network's weights and biases using batch gradient
     * descent on a batch (subset) of the training data.
     * 
     * @param batch a sub-set of the training data 
     * @param alpha the training rate.
     */
    public void updateWithBatch(List<ArrayList<Matrix>> batch, double alpha, 
                                double lambda, int n) {
        
        // Create matrix arrays to store cumulative deltas.
        Matrix[] weightUpdates = new Matrix[layers.length];
        Matrix[] biasUpdates = new Matrix[layers.length];
        for (int l = 0; l < layers.length; l++) {
            weightUpdates[l] = new Matrix(layers[l].getWeights().getRows(), 
                    layers[l].getWeights().getCols());
            biasUpdates[l] = new Matrix(layers[l].getBias().getRows(), 1);
        }
        
        // Feed each annotation in the batch through the network.
        for (ArrayList<Matrix> annotation : batch) {
            Matrix X = annotation.get(0);
            Matrix Y = annotation.get(1);
            forwardProp(X);
            Matrix[] deltas = backProp(Y);
            for (int i = 0; i < layers.length; i++) {
                weightUpdates[i] = Matrix.add(weightUpdates[i], deltas[2 * i]);
                biasUpdates[i] = Matrix.add(biasUpdates[i], deltas[2 * i+1]);
            }
        }
        
        // update with cumulative deltas, taking one large step of gradient descent.
        for (int j = 0; j < layers.length; j++) {
            layers[j].setWeights(
                    Matrix.subtract(
                    // update with regularization constant 1 - (αλ/n)
                    Matrix.multiply(layers[j].getWeights(), (1 - ((alpha * lambda) / n))), 
                    Matrix.multiply(weightUpdates[j], (alpha)))
                            
                            //layers[j].getWeights(), 
                            //Matrix.multiply(weightUpdates[j], alpha))
            );
                    
            layers[j].setBias(Matrix.subtract(layers[j].getBias(),
                    Matrix.multiply(biasUpdates[j], (alpha))));
        }
    }

    /**
     * Feeds the features of a single annotation through the network.
     *
     * @param inputs a utils.Matrix containing the attributes of a single annotation.
     * @return the final activation layer (output) of the network.
     */
    public Matrix forwardProp(Matrix inputs) {
        this.activations[0] = inputs;
        Matrix activation = inputs;
        for (int i = 0; i < layers.length; i++) {
            Matrix z = Matrix.add(Matrix.dotProduct(layers[i].getWeights(), activation), 
                    layers[i].getBias());
            this.zl[i] = z;
            activation = Matrix.map(SIGMOID, z); 
            this.activations[i + 1] = activation;
        }
        return activation;
    }

    /**
     * Back propagate an error through the network to generate the
     * derivatives for the cost function w.r.t the network's weights
     * and biases; the return value is a gradient vector used for updating
     * the network.
     * 
     * @param Y a single label
     * @return deltas; an array of gradient matrices.
     */
    public Matrix[] backProp(Matrix Y) {

        // A matrix array storing the paired derivatives for each layer
        // (layers[0].weights, layers[0].bias, layers[1].weights, 
        // layers[1].weights ...etc.)
        Matrix[] deltas = new Matrix[2 * layers.length];

        // First we compute the output error. 
        Matrix error = costDerivative(activations[activations.length -1], Y);

        // We calculate the gradients for the weights and biases that feed 
        // into the output layer (Y - A) . (σ(A) * (1 - σ(A))
        Matrix delta = Matrix.hadamardProduct(error, Matrix.map(SIGMOIDPRIME, zl[zl.length -1])); 

        // gradients for the weights that feed into the output layer.
        deltas[deltas.length -2] =
                Matrix.dotProduct(delta, Matrix.transpose(activations[activations.length-2]));

        // Gradients for the biases that feed into the output layer.
        deltas[deltas.length -1] = delta;

        // Backward pass through the remaining layers.
        for (int i = layers.length -2; i >= 0; i--) {

            // Calculate the error (wl+1)T . σl+1) -> delta = error ⊙ σ′(zl)
            error = Matrix.dotProduct(Matrix.transpose(layers[i + 1].getWeights()), delta);
            delta = Matrix.hadamardProduct(error, Matrix.map(SIGMOIDPRIME, zl[i])); 

            deltas[2 * i] = Matrix.dotProduct(delta, Matrix.transpose(activations[i]));
            deltas[2 * i + 1] = delta;
        }
        return deltas;
    }

    /**
     * Computes the L2 cost for a single prediction (yHat - y)^2
     * 
     * @param yHat output from the network.
     * @param y label associated with yHat.
     * @return l2 cost function output for a single annotation. 
     */
    public static double meanSquareError(Matrix yHat, Matrix y) {
        Matrix error = Matrix.subtract(yHat, y);
        Matrix errorSquared = Matrix.hadamardProduct(error, error);
        return 0.5 * Matrix.sum(errorSquared); 
    }

    /**
     * Computes the cross entropy cost function for a single
     * prediction: ∑ y(log a) + (1 - y)log(1 - a)
     * 
     * @param yHat output from the network.
     * @param y label associated with yHat.
     * @return cross entropy cost function output for a single annotation.
     */
    public static double crossEntropyCost(Matrix yHat, Matrix y) {
        
        double cost = 0.0;
        for (int i = 0; i < y.getRows(); i++) {
            for (int j = 0; j < y.getCols(); j++) {
                
                cost += -(y.getData()[i][j] * Math.log(yHat.getData()[i][j]) + 
                        (1 - y.getData()[i][j]) * Math.log(1 - yHat.getData()[i][j])); 
            }
        }
        return cost;
    }

    /**
     * Wrapper method to indicate where the cost function derivative
     * is applied in the backpropagation algorithm. 
     *
     * @param yHat: the output activation layer of the network.
     * @param y: a label corresponding the input which generated a.
     * @return the derivative of the mean squares cost function.
     */
    public Matrix costDerivative(Matrix yHat, Matrix y) {
        return Matrix.subtract(yHat, y);
    }

    /**
     * Evaluates the performance of the network's overall accuracy 
     * (the percentage of relevant results correctly classified).
     * 
     * @param dataSet preprocessed data; an arraylist of subarray 'annotations'.
     * @return a double value representing the percentage of correct predictions.
     */
    public double evaluateAccuracy(ArrayList<ArrayList<Matrix>> dataSet) {
        double correct = 0;

        for (ArrayList<Matrix> annotation : dataSet) {
            Matrix X = annotation.get(0);
            int y = Matrix.argMaxRow(annotation.get(1));

            int yHat = Matrix.argMaxRow(forwardProp(X));
            if (y == yHat) {
                correct++;
            }
        }
        return (correct / dataSet.size());
    }

    /**
     * Computes cost function of the network.
     * 
     * @param dataset preprocessed data; an arraylist of subarray 'annotations'.
     * @param lambda regularization constant.
     * @return the cost function; either mean square or cross entropy.
     */
    public double evaluateCost(ArrayList<ArrayList<Matrix>> dataset, double lambda) {
        double cost = 0;

        for (ArrayList<Matrix> annotation : dataset) {
            Matrix X = annotation.get(0);
            Matrix Y = annotation.get(1);
            Matrix yHat = forwardProp(X);
            
            //cost += meanSquareCost(yHat, Y) / dataset.size();
            cost += crossEntropyCost(yHat, Y) / dataset.size();
        }

        double wSquaredSum = 0.0;
        for (Layer layer : this.layers) {
            double[][] layerData = layer.getWeights().getData();
            Matrix w = new Matrix(layerData);
            wSquaredSum =+ Matrix.sum(Matrix.hadamardProduct(w, w));
        }
        
        // Finally we add our regularization term to the cost (cost + λ/2n∑ w^2)
        cost += 0.5 * ((lambda / (dataset.size())) * wSquaredSum);
        return cost;
    }

    /**
     * Outputs the predictions for the test set.
     * 
     * @param testSet preprocessed data; an arraylist of subarray 'annotations',
     * each annotation contains an attribute matrix and its corresponding label matrix.
     */
    public int[][] predict(ArrayList<ArrayList<Matrix>> testSet) {
   
        // row 0: testLabels, row 1: corresponding predictions
        int[][] results = new int[2][testSet.size()];
        
        for (int i = 0; i < testSet.size(); i++) {
            Matrix X = testSet.get(i).get(0);
            int y = Matrix.argMaxRow(testSet.get(i).get(1)); 
            results[0][i] = y;

            int yHat = Matrix.argMaxRow(forwardProp(X));
            results[1][i] = yHat;
        }
        return results;
    }

    /**
     * Layers treated separately for future extensibility.
     * Weights and biases cannot be accessed globally.
     */
    public static class Layer implements Serializable{
        /**
         * A matrix containing the values of the weights
         * feeding into the layer.
         */
        private Matrix weights;

        /**
         * A matrix containing the biases for each node in the layer.
         */
        private Matrix bias;

        /**
         * Constructs a new layer in the neural network.
         *
         * @param inputs:  the number of nodes in the previous layer.
         * @param outputs: the number of nodes in the layer itself.
         */
        public Layer(int inputs, int outputs) {
            this.weights = weightInitializer(outputs, inputs);
            this.bias = weightInitializer(outputs, 1);
        }

        /**
         * @return the weights feeding into the output nodes in a given layer.
         */
        public Matrix getWeights() {
            return this.weights;
        }

        /**
         * @param weights new weights for the layer.
         */
        public void setWeights(Matrix weights) {
            this.weights = weights;
        }

        /**
         * @return the biases feeding into the output nodes in a given layer.
         */
        public Matrix getBias() {
            return this.bias;
        }

        /**
         * @param bias new bias for the layer.
         */
        public void setBias(Matrix bias) {
            this.bias = bias;
        }

        /**
         * Returns a matrix of randomly generated Gaussian values 
         * with a mean 0.0 and a standard deviation of 1/√n (where
         * n is the number of input weights).
         *
         * @param rows number of specified rows for the matrix.
         * @param cols number of specified columns for the matrix.
         * @return a new rows x cols matrix.
         */
        public Matrix weightInitializer(int rows, int cols) {
            Random rand = new Random();

            Matrix product = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {

                    product.getData()[i][j] = rand.nextGaussian() / Math.sqrt(cols);
                }
            }
            return product;
        }
    }
}
