package utils;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class DataPrepTest {
    
    //----------------------------------------------------------------
    // samples from setA, used for both unit and integration testing, are
    // labeled as follows {triangle: 0, moon: 1},
    // data split {training : validation : test} -> {5600 : 1200 : 1200}
    //-----------------------------------------------------------------

    // data arrays used for various tests
    private double[][] samples;
    
    private ArrayList<ArrayList<Matrix>> samplesAsVectors;

    @BeforeEach
    public void init() throws IOException {
        
        samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/test/setA/test.dat"));
        
        samplesAsVectors = DataPrep.vectorize(samples, 2);
        
    }
    
    //------load method-------
    
    /**
     * Testing load method; uploading the correct number of data arrays from
     * the training, validation, and test splits.
     */
    @Test
    public void loadTest1() throws IOException {
        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/test/setA/train.dat"));
        
        int expected = 5600;
        int actual = samples.length;
        
        assertEquals(expected, actual);
    }

    @Test
    public void loadTest2() throws IOException {
        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/test/setA/validation.dat"));

        int expected = 1200;
        int actual = samples.length;

        assertEquals(expected, actual);
    }
    
    
    @Test
    public void loadTest3() throws IOException {
        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/test/setA/test.dat"));

        int expected = 1200;
        int actual = samples.length;

        assertEquals(expected, actual);
    }

    /**
     * Given a faulty file path, load() should throw a IOException
     */
    @Test
    public void loadTest4()  {
        
        assertThrows(IOException.class, () -> {
            DataPrep.loadData(new File(System.getProperty("user.dir")
                    + "/test/dummySet/validation.dat"));
        });
    }
    
    //-----vectorize method-----

    /**
     * Check the resulting vectorized data set where each of the 1200
     * data points is represented as a pair of column matrices (X, Y).
     */
    @Test
    public void vectorizeTest1() throws IOException {

        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/Test/setA/train.dat"));

        ArrayList<ArrayList<Matrix>> vectors = DataPrep.vectorize(samples, 2);
        
        // assert the size of each annotation
        int annotationSize = 2;


        for (ArrayList<Matrix> vector : vectors) {
            assertEquals(vector.size(), annotationSize);
        }
        
        
        // assert the number of {X, Y} data points
        int expected = 5600;
        int actual = vectors.size();
        
        assertEquals(expected, actual);
    }

    /**
     * Same as above; done with validation set
     * 
     * @throws IOException (loadData)
     */
    @Test
    public void vectorizeTest2() throws IOException {

        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/Test/setA/validation.dat"));

        ArrayList<ArrayList<Matrix>> vectors = DataPrep.vectorize(samples, 2);

        // assert the size of each annotation
        int annotationSize = 2;


        for (ArrayList<Matrix> vector : vectors) {
            assertEquals(vector.size(), annotationSize);
        }


        // assert the number of {X, Y} data points
        int expected = 1200;
        int actual = vectors.size();

        assertEquals(expected, actual);
    }

    /**
     * Same as above; done with test set.
     * 
     * @throws IOException (loadData)
     */
    @Test
    public void vectorizeTest3() throws IOException {

        double[][] samples = DataPrep.loadData(new File(System.getProperty("user.dir")
                + "/Test/setA/test.dat"));

        ArrayList<ArrayList<Matrix>> vectors = DataPrep.vectorize(samples, 2);

        // assert the size of each annotation
        int annotationSize = 2;


        for (ArrayList<Matrix> vector : vectors) {
            assertEquals(vector.size(), annotationSize);
        }


        // assert the number of {X, Y} data points
        int expected = 1200;
        int actual = vectors.size();

        assertEquals(expected, actual);
    }
    
    /**
     * Checking the dimensions of the resulting vectors
     */
    @Test
    public void vectorizeTest4() {
        
        // feature vector X
        int xRows = 784;
        int xCols = 1;

        // label vector Y
        int yRows = 2;
        int yCols = 1;
        
        for (int i = 0; i < samplesAsVectors.size(); i++) {

            // assert dimensions of X
            assertEquals(samplesAsVectors.get(i).get(0).getRows(), xRows);
            assertEquals(samplesAsVectors.get(i).get(0).getCols(), xCols);

            // assert dimensions of Y
            assertEquals(samplesAsVectors.get(i).get(1).getRows(), yRows);
            assertEquals(samplesAsVectors.get(i).get(1).getCols(), yCols);
         }
    }
    
    //------split------

    /**
     * Check dimensions of resulting folds; samples = 1200 therefore
     * a 10 fold split -> 120 data arrays per fold
     */
    @Test
    public void splitTest1() {
        ArrayList<double[][]> folds = DataPrep.split(samples, 10);
        
        int expectedFoldSize = 120;
        for (int i = 0; i < folds.size(); i++) {
            assertEquals(folds.get(i).length, expectedFoldSize);
        }
    }

    /**
     * Checking contents of the split
     */
    @Test
    public void splitTest2() {
        double[][] rows = {
                {3.0, 4.0, 5.0},
                {9.0, 10.0, 11.0},
                {5.0, 82.3, 13.0},
                {5.0, 77.4, 9}
        };
        
        double[][] splitOne = {
                {3.0, 4.0, 5.0},
                {9.0, 10.0, 11.0}   
        };

        double[][] splitTwo = {
                {5.0, 82.3, 13.0},
                {5.0, 77.4, 9}
        };

        ArrayList<double[][]> folds = DataPrep.split(rows, 2);

        for (int i = 0; i < splitOne.length; i++) {
            assertTrue(Arrays.equals(splitOne[i], folds.get(0)[i])); 
        }

        for (int i = 0; i < splitOne.length; i++) {
            assertTrue(Arrays.equals(splitTwo[i], folds.get(01)[i]));
        }
    }

    /**
     * Cannot split training set cleanly into 9 folds (1200 / 9 = 133.4)
     */
    @Test
    public void splitTest3() {
        assertThrows(IllegalArgumentException.class, () -> {
            DataPrep.split(samples, 9);
        });
    }
    
    //-----oneHotEncode-------

    /**
     * Cannot create a vector with a negative label
     */
    @Test
    public void OHETest1() {
        assertThrows(IllegalArgumentException.class, () -> {
            DataPrep.oneHotEncode(-9, 10);
        });
    }

    /**
     * label number cannot exceed output (rows for the vector)
     */
    @Test
    public void OHETest2() {
        assertThrows(IllegalArgumentException.class, () -> {
            DataPrep.oneHotEncode(40, 3);
        });
    }

    /**
     * Test method creates a binary vector, with the index corresponding to the label
     * equalling 1 and all other indices equalling 0.
     */
    @Test
    public void OHETest3() {
        int label = 3;
        
        Matrix encoded = DataPrep.oneHotEncode(3, 5);
        
        for (int i = 0; i < encoded.getRows(); i++) {
            for (int j = 0; j < encoded.getCols(); j++) {
                if (encoded.getData()[i][j] != encoded.getData()[label][0]) {
                    assertEquals(0, encoded.getData()[i][j]);
                }
                assertEquals(encoded.getData()[label][0], 1);
            }
        }
    }
    
    //-----append-----
    // Note: only ever used in the context of split
    // so testing for variable sized rows is not necessary.

    /**
     * Appending to 2 2d arrays
     */
    @Test
    public void appendTest1() {
        double[][] a = {
                {3, 4, 5},
                {6, 7, 8}
        };
        double[][] b = {
                {9, 10, 11},
                {12, 13, 14}
        };
        
        double[][] expected = {
                {3.0, 4.0, 5.0}, 
                {6.0, 7.0, 8.0},
                {9.0, 10.0, 11.0},
                {12.0, 13.0, 14.0}
        };
        
        double[][] actual = DataPrep.append(a, b);
        assertTrue(Arrays.deepEquals(expected, actual));
    }
    
    //----append----
    
    /**
     * Adding duplicate arrays
     */
    @Test
    public void appendTest2() {
        double[][] a = {
                {3, 4, 5},
                {6, 7, 8},
        };
        double[][] b = {
                {3, 4, 5},
                {6, 7, 8}
        };

        double[][] expected = {
                {3, 4, 5},
                {6, 7, 8},
                {3, 4, 5},
                {6, 7, 8}
        };

        double[][] actual = DataPrep.append(a, b);
        assertTrue(Arrays.deepEquals(expected, actual));
    }
    
    /**
     * Single row 2d arrays
     */
    @Test
    public void appendTest3() {
        double[][] a = {
                {3, 4, 5}
        };
        double[][] b = {
                {9, 10, 11},
        };

        double[][] expected = {
                {3.0, 4.0, 5.0},
                {9.0, 10.0, 11.0},
        };

        double[][] actual = DataPrep.append(a, b);
        assertTrue(Arrays.deepEquals(expected, actual));
    }
}