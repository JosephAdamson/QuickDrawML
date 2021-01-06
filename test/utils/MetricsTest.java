package utils;

import static org.junit.jupiter.api.Assertions.*;

class MetricsTest {

    /**
     * k-nn accuracy evaluation; zero percent correct.  
     */
    public void accuracyTest1() {

        int[][] results = {
                {0, 0, 0},
                {1, 1, 1}
        };

        double expected = 0.0;
        double actual = Metrics.modelAccuracy(results);

        assertEquals(expected, actual);
    }

    /**
     * 25 percent correct.
     */
    public void accuracyTest2() {

        int[][] results = {
                {0, 0, 0, 0},
                {1, 1, 1, 0}
        };

        double expected = 0.25;
        double actual = Metrics.modelAccuracy(results);

        assertEquals(expected, actual);
    }
    
    public void confusionTest1() {
        
        // three classes
        int[] labels =      {1, 1, 3, 2, 3, 3, 3, 3, 3, 3, 1, 1, 3, 2, 1, 3, 2, 2};
        int[] predictions = {1, 1, 3, 2, 3, 2, 2, 1, 3, 3, 1, 1, 3, 1, 1, 3, 2, 3};
        
        
    }

}