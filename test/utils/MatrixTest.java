package utils;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import utils.Matrix;

import java.util.function.Function;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {
    
    private Matrix allZeros;
    private Matrix allZeros2;
    private Matrix m1;
    private Matrix m2;
    private Matrix m3;
    private Matrix m4;
    private Matrix m5;
    private Matrix m6;
    private Matrix m7;
    private Matrix m8;
    
    @BeforeEach
    public void init() {
        allZeros = new Matrix(3, 4);
        allZeros2 = new Matrix(4, 3);
        double[][] data1 = {
                {11.0, 5.0, 19.0, 3.6},
                {7.0, 6.0, 2.0, 2.0},
                {55.0, 3.0, 9.0, 1.0}
        };
        m1 = new Matrix(data1);
        double[][] data2 = {
                {8.0, 7.0, 14.5, 3.3},
                {7.0, 5.7, 9.3, 4.0},
                {36.8, 6.2, 4.0, 4.0}
        };
        m2 = new Matrix(data2);
        double[][] data3 = {
                {6.2, 7.8, 3.0},
                {12.0, 11.0, 10.0},
                {8.0, 8.0, 4.5}
        };
        m3 = new Matrix(data3);
        double[] data4 = {1.3, 2.0, 3.4};
        m4 = new Matrix(data4);
        double[][] data5 = {
                {0.4},
                {5.0},
                {6.7}
        };
        m5 = new Matrix(data5);
        double[] data6 = {36};
        m6 = new Matrix(data6);
        double[][] data7 = {
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0}
        };
        m7 = new Matrix(data7);
        double[][] data8 = {
                {0.0, 1.0, 0.0},
                {0.0, 2.0, 0.0},
                {0.0, 0.0, 3.0},
                {1.0, 0.0, 0.0} 
        };
        m8 = new Matrix(data8);
        double[][] data9 = {
                {6.2, -7.8, 3.0},
                {12.0, 11.0, -10.0},
                {8.0, 8.0, 4.5}
        };
    }
    
    //--------------Constructors---------------

    /**
     * rows of input data must be of equal length.
     */
    @Test
    public void constructorTest() {
        double[][] data = {
                {2, 3, 5},
                {7},
                {4, 5, 6, 2}
        };
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix malformed = new Matrix(data);
        });
    }
    
    //---------arithmetic operations-----------

    /**
     * Scalar addition.
     */
    @Test
    public void addTest1() {
        double[][] data = {
                {13.2, 7.2, 21.2, 5.8},
                {9.2, 8.2, 4.2, 4.2},
                {57.2, 5.2, 11.2, 3.2}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.add(m1, 2.2);
        assertTrue(expected.equals(actual));
    }

    /**
     * Scalar addition; negative scalar.
     */
    @Test
    public void addTest2() {
        double[][] data = {
                {1.8, -4.2, 9.8, -5.6},
                {-2.2, -3.2, -7.2, -7.2},
                {45.8, -6.2, -0.2, -8.2}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.add(m1, -9.2);
        assertTrue(expected.equals(actual));
    }

    /**
     * Scalar addition; zero scalar.
     */
    @Test
    public void addTest3() {
       Matrix actual = Matrix.add(m1, 0);
       assertTrue(m1.equals(actual));
    }

    /**
     * Elementwise addition.
     */
    @Test
    public void addTest4() {
        double[][] data = {
                {19.0, 12.0, 33.5, 6.9},
                {14, 11.7, 11.3, 6.0},
                {91.8, 9.2, 13.0, 5.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.add(m1, m2);
        assertTrue(expected.equals(actual));
    }

    /**
     * Adding zero array, elementwise addition.
     */
    @Test
    public void addTest5() {
        Matrix actual = Matrix.add(m1, allZeros);
        assertTrue(m1.equals(actual));
    }

    /**
     * A doesn't have same dimensions as B; cannot add A and B.
     */
    @Test
    public void addTest6() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.add(m1, m3);
        });
    }

    /**
     * Scalar subtraction
     */
    @Test
    public void subtractTest1() {
        double[][] data = {
                {7.9, 1.9, 15.9, 0.5},
                {3.9, 2.9, -1.1, -1.1},
                {51.9, -0.1, 5.9, -2.1}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.subtract(m1, 3.1);
        assertTrue(expected.equals(actual));
    }

    /**
     * Scalar subtraction; negative scalar.
     */
    @Test
    public void subtractTest2() {
        double[][] data = {
                {14.6, 8.6, 22.6, 7.2},
                {10.6, 9.6, 5.6, 5.6},
                {58.6, 6.6, 12.6, 4.6}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.subtract(m1, -3.6);
        assertTrue(expected.equals(actual));
    }

    /**
     * Scalar addition; zero scalar.
     */
    @Test
    public void subtractTest3() {
        Matrix actual = Matrix.subtract(m1, 0);
        assertTrue(m1.equals(actual));
    }

    /**
     * Elementwise subtraction.
     */
    @Test
    public void subtractTest4() {
        double[][] data = {
                {3.0, -2.0, 4.5, 0.3},
                {0.0, 0.3, -7.3, -2.0},
                {18.2, -3.2, 5.0, -3.0}
        }; 
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.subtract(m1, m2);
        assertTrue(expected.equals(actual));
    }

    /**
     * Subtraction zero array; elementwise subtraction.
     */
    @Test
    public void subtractTest5() {
        Matrix actual = Matrix.subtract(m1, allZeros);
        assertTrue(m1.equals(actual));
    }

    /**
     * A doesn't have same dimensions and B; cannot subtract B form A.
     */
    @Test
    public void subtractTest6() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.subtract(m1, m3);
        });
    }

    /**
     * [A.rows x A.cols] . [B.rows x B.cols] = [A.rows x B.cols]
     *  -> [1 x 3] . [3 x 1] = [1 x 1]
     */
    @Test
    public void dotTest1() {
        double[] data = {33.3};
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.dotProduct(m4, m5);
        assertTrue(expected.equals(actual));
    }

    /**
     * [A.rows x A.cols] . [B.rows x B.cols] = [A.rows x B.cols]
     * -> [3 x 1] . [1 x 3] = [3 x 3]
     */
    @Test
    public void dotTest2() {
        double[][] data = {
                {0.52, 0.8, 1.36},
                {6.5, 10.0, 17.0},
                {8.71, 13.4, 22.78}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.dotProduct(m5, m4);
        assertTrue(expected.equals(actual));
    }

    /**
     * Dot with zero matrix.
     */
    @Test
    public void datTest3() {
        double[][] data = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.dotProduct(m1, allZeros2);
        assertTrue(expected.equals(actual));
    }

    /**
     * A.cols != B.rows
     */
    @Test
    public void dotTest4() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.dotProduct(m1, m3);
        });
    }

    /**
     * Normal case
     */
    @Test
    public void hadamardTest1() {
        double[][] data = {
                {88.0, 35.0, 275.5, 11.88}, 
                {49.0, 34.2, 18.6, 8.0},
                {2024.0, 18.6, 36.0, 4.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.hadamardProduct(m1, m2);
        assertTrue(expected.equals(actual));
    }

    /**
     * Multiplied with a matrix of zeros.
     */
    @Test
    public void hadamardTest2() {
        double[][] data = {
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 0.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.hadamardProduct(m1, allZeros);
        assertTrue(expected.equals(actual));
    }

    /**
     * Two matrices that are the same; a matrix a squares
     */
    @Test
    public void hadamardTest3() {
        double[][] data = {
                {0.16},
                {25.0},
                {44.89}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.hadamardProduct(m5, m5);
        assertTrue(expected.equals(actual));
    }

    /**
     * Dimensions of A and B do not match
     */
    @Test
    public void hadamardTest4() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.hadamardProduct(m1, m3);
        });
    }

    /**
     * Scalar multiplication.
     */
    @Test
    public void multiplyTest1() {
        double[][] data = {
                {81.95, 37.25, 141.55, 26.82},
                {52.15, 44.7, 14.9, 14.9},
                {409.75, 22.35, 67.05, 7.45}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.multiply(m1, 7.45);
        assertTrue(expected.equals(actual));
    }

    /**
     * Negative scalar multiplication.
     */
    @Test
    public void multiplyTest2() {
        double[][] data = {
                {-0.011, -0.005, -0.019, -0.0036},
                {-0.007, -0.006, -0.002, -0.002},
                {-0.055, -0.003, -0.009, -0.001}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.multiply(m1, -0.001);
        assertTrue(expected.equals(actual));
    }

    /**
     * Scalar multiplication with zeros
     */
    @Test
    public void multiplyTest3() {
        double[][] data = {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.multiply(m1, 0);
        assertTrue(expected.equals(actual));
    }

    //-----------misc. methods------------
    
    /**
     * Randomize cannot have negative parameters.
     */
    @Test
    public void randomizeTest() {
        assertThrows(NegativeArraySizeException.class, () -> {
            Matrix.randomize(3, -3);
        }); 
    }

    /**
     * total elements of m1 != rows * cols
     */
    @Test
    public void reshapeTest1() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.reshape(m1,3, 6);
        });
    }

    /**
     * Total elements of m1 are not divisible by rows (5) 
     */
    @Test
    public void reshapeTest2() {
        assertThrows(IllegalArgumentException.class, () -> {
            Matrix.reshape(m1,5, 6);
        });
    }

    /**
     * Negative dimensions.
     */
    @Test
    public void reshapeTest3() {
        assertThrows(NegativeArraySizeException.class, () -> {
            Matrix.reshape(m1,-2, -6);
        });
    }

    /**
     * Reshape m1 (3 x 4) -> (6 x 2)
     */
    @Test
    public void reshapeTest4() {
        double[][] data = {
                {11.0, 5.0},
                {19.0, 3.6},
                {7.0, 6.0},
                {2.0, 2.0},
                {55.0, 3.0},
                {9.0, 1.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.reshape(m1, 6, 2);
        assertTrue(expected.equals(actual));
    }
    
    @Test
    public void transposeTest1() {
        double[][] data = {
                {11.0, 7.0, 55.0},
                {5.0, 6.0, 3.0},
                {19.0, 2.0, 9.0},
                {3.6, 2.0, 1.0}
        };
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.transpose(m1);
        assertTrue(expected.equals(actual));
    }

    /**
     * Transposing a singleton should not effect its state.
     */
    @Test
    public void transposeTest2() {
        double[] data = {36};
        Matrix expected = new Matrix(data);
        Matrix actual = Matrix.transpose(m6);
        assertTrue(expected.equals(actual));
    }

    /**
     * <pre>
     *  [[11.0, 5.0, 19.0, 3.6]
     *   [7.0, 6.0, 2.0, 2.0]
     *  [55.0, 3.0, 9.0, 1.0]] <- max elem in row 2 
     * </pre>
     */
    @Test
    public void argMaxRow1() {
        double expected = 2;
        double actual = Matrix.argMaxRow(m1);
        assertEquals(expected, actual);
    }

    /**
     * Equal elements in separate rows; first instance returned
     */
    @Test
    public void argMaxRow2() {
       double expected = 0;
       double actual = Matrix.argMaxRow(m7);
       assertEquals(expected, actual);
    }

    /**
     * Greater element in each subsequent row.
     */
    @Test
    public void argMaxRow3() {
        double expected = 2;
        double actual = Matrix.argMaxRow(m8);
        assertEquals(expected, actual);
    }

    /**
     * Summing all elements in m1.
     */
    @Test
    public void sumTest1() {
        double expected = 123.6;
        double actual = Matrix.sum(m1);
        assertEquals(expected, actual);
    }

    /**
     * Summing a singleton.
     */
    @Test
    public void sumTest2() {
        double expected = 36;
        double actual = Matrix.sum(m6);
        assertEquals(expected, actual);
    }

    /**
     * Simple function applied.
     */
    @Test
    public void mapTest() {
        Function<Double, Double> simple = (x) -> (x + 5) * 2;
        double[][] data3 = {
                {22.4, 25.6, 16.0},
                {34.0, 32.0, 30.0},
                {26.0, 26.0, 19.0}
        };
        Matrix expected = new Matrix(data3);
        Matrix actual = Matrix.map(simple, m3);
        assertTrue(expected.equals(actual));
    }
}