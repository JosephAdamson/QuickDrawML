package models;

import neuralnetwork.NeuralNetwork;
import utils.DataPrep;
import utils.Matrix;
import utils.Metrics;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class NeuralNetworkMultiClass {
    public static void main(String[] args) throws IOException {

        //----------------------network hyperParameters------------------------

        int epochs = 50;

        // learning rate.
        double alpha = 0.01;

        // regularization constant.
        double lambda = 0.005;

        // mini batch size.
        int batch = 32;

        //---------------------network training--------------------------

        /*NeuralNetwork nn = new NeuralNetwork(784, 90, 5);

        ArrayList<ArrayList<Matrix>> training =
                DataPrep.vectorize(DataPrep.loadData(new File(System.getProperty("user.dir")
                        + "/data/setB/train.dat")), nn.getOutputNodes());

        ArrayList<ArrayList<Matrix>> validation =
                DataPrep.vectorize(DataPrep.loadData(new File(System.getProperty("user.dir")
                        + "/data/setB/validation.dat")), nn.getOutputNodes());

        ArrayList<ArrayList<Matrix>> testing =
                DataPrep.vectorize(DataPrep.loadData(new File(System.getProperty("user.dir")
                        + "/data/setB/test.dat")), nn.getOutputNodes());

        double[][] results = nn.mbgd(training, epochs, batch,
                alpha, lambda, validation);


        Metrics.plotNetworkResults(epochs, results);
        Metrics.confusionMatrix(nn.predict(testing));
        nn.saveNetwork();*/

        //--------------------load optimized network--------------------------


        NeuralNetwork optimizedNN = NeuralNetwork.loadNetwork(new File(System.getProperty("user.dir")
                + "/data/networkModel.dat"));
        ArrayList<ArrayList<Matrix>> testing =
                DataPrep.vectorize(DataPrep.loadData(new File(System.getProperty("user.dir")
                        + "/data/setB/test.dat")), 5);

        System.out.print("\n===Neural network confusion matrix===");
        Metrics.confusionMatrix(optimizedNN.predict(testing));

    }
}