package knearestneighbours;

import utils.DataPrep;
import utils.Metrics;
import java.util.*;

/**
 * K nearest neighbours implementation with cross validation. Used to classify 
 * a subset of images from the Google Quick, Draw! dataset.
 * 
 * @author Joseph Adamson
 */
public class KNearestNeighbours {
    
    /**
     * Simple constructor for k nearest neighbour classifier.
     */
    public KNearestNeighbours() {}
    
    /**
     * Compute the Euclidean distance between two features.
     * 
     * @param XPrime a test feature set.
     * @param X a training feature set.
     * @return the distance between vectors X and Y: √(Y - X)^2
     */
    public static double euclideanDistance(double[] XPrime, double[] X) {
        if (X.length != XPrime.length) {
            throw new IllegalArgumentException("Dimensions of X and XPrime must match");
        }
        
        double sumDifference = 0.0;
        
        // last index is the label and therefore omitted from the calculation.
        for (int i = 0; i < XPrime.length -1; i++) {
            sumDifference += Math.pow((XPrime[i] - X[i]), 2);
        }
        return Math.sqrt(sumDifference);
    }

    /**
     * The method computes the distance between Y (a single test feature set)
     * and each feature set in the training data.
     * 
     * @param training a 2d array where each row contains 785 values; 0 - 784 contains normalized
     * pixel values with the last element denoting the label of the image. 
     * @param testing a single test feature set.
     * @return a 2d array where each row corresponds to a test feature set Y
     * and each column corresponds to the distance between Y and a given training 
     * feature set. 
     */
    public Distance[] getDistances(double[][] training, double[] testing) {
        Distance[] distances = new Distance[training.length];
        
        for (int i = 0; i < training.length; i++) {
            double respectiveDistance = euclideanDistance(testing, training[i]); 
            distances[i] = new Distance(respectiveDistance, training[i][training[i].length -1]);
        }
        Arrays.sort(distances, new DistanceComp());
        return distances;
    }
    
    /**
     * Returns the most frequently appearing neighbour for a given
     * group of nearest neighbours by majority vote.
     *
     * @param nearestNeighbours an array list containing the respective distances between each 
     * training instance X for a given test instance Y.
     * @param k the number of neighbours used for the prediction.
     * @return the dominant neighbour (the most frequently occurring) in a selected
     * group of k neighbours.
     */
    public int findBestNeighbour(Distance[] nearestNeighbours, int k) {
        Map<Double, Integer> frequencies = new HashMap<>();

        for (int i = 0; i < k; i++) {
            Integer freq = frequencies.get(nearestNeighbours[i].getLabel());
            frequencies.put(nearestNeighbours[i].getLabel(), freq == null ? 1 : freq + 1);
        }
        
        // returns closest distance (k=1) in event of a tie
        Set<Integer> values = new HashSet<>(frequencies.values());
        if (values.size() == 1) {
            return (int)nearestNeighbours[0].getLabel();
        } else {

            Map.Entry<Double, Integer> max = frequencies.entrySet().iterator().next();
            for (Map.Entry<Double, Integer> entry : frequencies.entrySet()) {
                if (entry.getValue() > max.getValue()) {
                    max = entry;
                }
            }
            return max.getKey().intValue();
        }
    }

    /**
     * Returns the accuracy rating for total predictions on a given test set. 
     * 
     * @param training a 2d array where each row contains 785 values; 0 - 784 contains normalized
     * pixel values with the last denoting the label of the image.
     * @param testing see above.
     * @param k the number of neighbours used for the prediction.            
     */
    public int[][] predict(double[][] training, double[][] testing, int k) {
       
        int[][] results = new int[testing.length][testing.length];

        training = DataPrep.shuffleData(training);
        testing = DataPrep.shuffleData(testing);
        
        for (int i = 0; i < testing.length; i++) {
            Distance[] distances = getDistances(training, testing[i]);
            results[0][i] = (int)testing[i][testing[i].length -1];
            results[1][i] = findBestNeighbour(distances, k);
        }
        return results;
    }

    /**
     * Computes the percentage of incorrect predictions for the test data;
     * the error rate.
     * 
     * @param labels labels from the test data
     * @param predictions predicted labels for the test data
     * @return a percentage value indication of the classifier's
     * error rate.
     */
    public double evaluateError(int[] labels, int[] predictions) {
        double errorSum = 0.0;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] != predictions[i]) {
                errorSum++;
            }
        }
        return (errorSum / labels.length);
    }

    /**
     * Training data is split up into n subsets or 'folds', for each iteration
     * in the range n (number of folds) subset i is held used as a validation set 
     * and tested against the remaining (n - 1) subsets which, when combined, act as 
     * the training data.
     *  
     * @param training a 2d array where each row contains 785 values; 0 - 784 contains normalized
     * pixel values with the last denoting the label of the image 
     * @param folds the number of subsets the training data will be split into.
     * @return an array of average accuracies for each k value (1 - 30) over the number
     * of folds.
     */
    public double[] kFoldCrossValidation(double[][] training, int folds) {
        training = DataPrep.shuffleData(training);
        ArrayList<double[][]> foldedData = DataPrep.split(training, folds);

        // to store final errors for each value of k in the range (1 - 30, exclusive)
        double[] errorsForK = new double[100];

        // 2d array used to store errors for each value of k (30 rows for k values 1 - 30)
        // each column corresponding to a error for that value for a fold.
        double[][] kAvg = new double[100][folds];

        // for each fold in the split data
        for (int i = 0; i < foldedData.size(); i++) {
            
            // separate validation fold
            double[][] validation = foldedData.get(i);
            
            // define the temp training (n - 1 folds)
            double[][] tempTraining = {};
            for (int j = 0; j < foldedData.size(); j++) {
                if (j != i) {
                    tempTraining = DataPrep.append(tempTraining, foldedData.get(j));
                }
            }
            
            // Get our distance table and extract validation labels.
            Distance[][] distanceLookup = new Distance[validation.length][tempTraining.length];
            int[] validationLabels = new int[validation.length];
            for (int row = 0; row < validation.length; row++) {
                distanceLookup[row] = getDistances(tempTraining, validation[row]);
                validationLabels[row] = (int)validation[row][validation[row].length -1];
            }
            
           /* for (int t = 0; t < distanceLookup.length; t++) {
                for(int y = 0; y < distanceLookup[t].length; y++) {
                    System.out.print(distanceLookup[t][y].getDistance() + ", ");
                }
                System.out.println();
            }*/
            
            // use distanceLookup to test every k value to save on compute time.
            int kRow = 0;
            for (int k = 1; k < 101; k++) {
                int[] predictions = new int[validation.length];
                for (int m = 0; m < distanceLookup.length; m++) {
                    predictions[m] = findBestNeighbour(distanceLookup[m], k);
                }
                double error = evaluateError(validationLabels, predictions);
                kAvg[kRow++][i] = error;
            }
        }
        
        // Compute error averages for each k value across folds.
        for (int row = 0; row < kAvg.length; row++) {
            double sumAvg = 0.0;
            for (int col = 0; col < kAvg[row].length; col++) {
               sumAvg += kAvg[row][col]; 
            }
            errorsForK[row] = sumAvg / kAvg[0].length;
        
        }
        return errorsForK;
    }
}
