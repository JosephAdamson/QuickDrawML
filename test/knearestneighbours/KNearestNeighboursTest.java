package knearestneighbours;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import utils.DataPrep;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class KNearestNeighboursTest {
    
    //----euclideanDistance----
    
    private double[] sample1;
    private double[] sample2;
    private double[] sample3;
    private double[] sample4;
    private double[] zeros1;
    private double[] zeros2;
    private double[] singleton1;
    private double[] singleton2;
    private double[] test;
    private double[][] training;
    private double[][] training2;
    private double[][] training3;
    private KNearestNeighbours knn;
    
    
    @BeforeEach
    public void init() {
        sample1 = new double[]{3.7, 5.3, 0.0, 3.2, 3.0};
        sample2 = new double[]{2.6, 4.9, 0.0, 0.0, 6.0};
        sample3 = new double[]{3.7, 5.3, 0.0, 3.1, 3.0};
        sample4 = new double[]{3.7, 5.3, 0.0, 3.0, 3.0};
        zeros1 = new double[]{0.0, 0.0, 0.0, 0.0, 5.0};
        zeros2 = new double[]{0.0, 0.0, 0.0, 0.0, 9.0};
        singleton1 = new double[]{6.0};
        singleton2 = new double[]{8.0};
        
        // Dummy test
        test = sample1;
        
        // Dummy train
        training = new double[][]{sample2, sample3, zeros1};
        
        // Dummy train; similar neighbours
        training2 = new double[][]{sample2, sample3, zeros1, sample4};
        
        // Dummy train; equal tie
        training3 = new double[][]{sample2, sample4, zeros1, zeros2};
        
        knn = new KNearestNeighbours();
        
    }

    /**
     * subtraction: sample2 < sample1
     */
    @Test
    public void EuclidTest1() {
        double expected = 3.4073450074801643;
        double actual = KNearestNeighbours.euclideanDistance(sample1, sample2);
        
        assertEquals(expected, actual);
    }

    /**
     * all zeros, labels ignored
     */
    @Test
    public void EuclidTest2() {
        double expected = 0.0;
        double actual = KNearestNeighbours.euclideanDistance(zeros1, zeros2);
        
        assertEquals(expected, actual);
    }

    /**
     * zeros1 - sample1 = negative result; positive due to squaring.
     */
    @Test
    public void EuclidTest3() {
        double expected = 7.212489168102785;
        double actual = KNearestNeighbours.euclideanDistance(zeros1, sample1);

        assertEquals(expected, actual);
    }

    /**
     *  singletons; technically data arrays with no dimensions (as last element
     *  is assigned for the label. Should result in zero.
     */
    @Test
    public void EuclidTest4() {
        double expected = 0.0;
        double actual = KNearestNeighbours.euclideanDistance(singleton1, singleton2);

        assertEquals(expected, actual);
    }

    /**
     * Cannot perform operation on arrays of different length.
     */
    @Test
    public void EuclidTest5() {
        assertThrows(IllegalArgumentException.class, () -> {
            KNearestNeighbours.euclideanDistance(sample1, singleton2);
        });
    }

    //----getDistances----

    /**
     * Testing the sorting of distances by pre-predicting labels.
     */
    @Test
    public void getDistance(){
        
        // labels
        double[] expected = {3, 6, 5};
        
        Distance[] actual = knn.getDistances(training, test);
        
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i].getLabel());
        }
    }

    /**
     * Finding a neighbour with k=3, two similar labels in the top 3
     */
    @Test
    public void findBestNeighbourTest1() {
        Distance[] result = knn.getDistances(training2, test);
        
        int expected = 3;
        int actual = knn.findBestNeighbour(result, 3);
        
        assertEquals(expected, actual);
    }

    /**
     * returns closest distance in the event of a tie
     */
    @Test
    public void findBestNeighbourTest2() {
        Distance[] result = knn.getDistances(training3, test);

        int expected = 3;
        int actual = knn.findBestNeighbour(result, 3);

        assertEquals(expected, actual);
    }
    
    @Test
    public void validationSplitTest() {
        double[][] samples = {
                {6.2, 7.8, 3.0},
                {12.0, 11.0, 10.0},
                {8.0, 8.0, 4.5},
                {8.0, 7.0, 14.5, 3.3},
                {7.0, 5.7, 9.3, 4.0},
                {36.8, 6.2, 4.0, 4.0}
        };
        
        ArrayList<double[][]> foldedData = DataPrep.split(samples, 3);
        for (int i = 0; i < 1; i++) {

            // separate validation fold
            double[][] validation = foldedData.get(i);

            // define the temp training (n - 1 folds)
            double[][] tempTraining = {};
            for (int j = 0; j < foldedData.size(); j++) {
                if (j != i) {
                    tempTraining = DataPrep.append(tempTraining, foldedData.get(j));
                }
            }

            for (int p = 0; p < tempTraining.length; p++) {
                System.out.println(Arrays.toString(tempTraining[p]));
            }
            System.out.println("\n");

            for (int w = 0; w < validation.length; w++) {
                System.out.println(Arrays.toString(validation[w]));
            }
                
        }
    }

}