package models;

import knearestneighbours.KNearestNeighbours;
import utils.DataPrep;
import utils.Metrics;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class KNNMultiClass {
    public static void main(String[] args) throws IOException {
        KNearestNeighbours knn = new KNearestNeighbours();
        
        /*double[][] training = DataPrep.append(DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/data/setB/train.dat")), DataPrep.loadData(new File(System.getProperty("user.dir")
                        + "/data/setB/validation.dat")));*/

        double[][] training = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/data/setB/train.dat"));
        
        double[][] testing = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/data/setB/test.dat"));
        
        //double[] errors = knn.kFoldCrossValidation(training, 10);
        System.out.print("Building model...\n\n===k-nn confusion matrix===");
        Metrics.confusionMatrix(knn.predict(training, testing, 4));
        
        /*Metrics.plotCrossValidationResults("Error Rate vs K Value", " k value", "error",
                "error", errors);*/
    }
}
