package neuralnetwork;

import utils.Matrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {
    
    private NeuralNetwork nn;
    private Matrix m1;
    private  Matrix m2;
    private Matrix zeros;
    
    @BeforeEach
    public void init(){
        nn = new NeuralNetwork(3, 6, 2);

        double[][] data1 = {
                {11.0, 5.0, 19.0, 3.6},
        };
        m1 = new Matrix(data1);

        double[][] data2 = {
                {8.0, 7.0, 14.5, 3.3},
        };
        m2 = new Matrix(data2);
        
        zeros = new Matrix(1, 4);
        
    }
    
    //-----constructor tests------

    /**
     * Zero neurons first layer.
     */
    @Test
    public void constructorTest1() {
        assertThrows(IllegalArgumentException.class, () -> {
            NeuralNetwork nn = new NeuralNetwork(3, 0);
        });
    }

    /**
     * later layer has zero neurons.
     */
    @Test
    public void constructorTest2() {
        assertThrows(IllegalArgumentException.class, () -> {
            NeuralNetwork nn = new NeuralNetwork(3, 3, 4, 0, 9);
        });
    }

    /**
     * Single layer network
     */
    @Test
    public void constructorTest3() {
        NeuralNetwork nn = new NeuralNetwork(3, 1);
        
        int expected = 2;
        int actual = nn.getLayerNumber();
        assertEquals(expected, actual);
    }

    /**
     * Multiple layers.
     */
    @Test
    public void constructorTest4() {
        NeuralNetwork nn = new NeuralNetwork(3, 1, 50, 9, 1000, 2);

        int expected = 6;
        int actual = nn.getLayerNumber();
        assertEquals(expected, actual);
    }
    
    //-----cost functions-----

    /**
     * Normal case
     */
    @Test
    public void MSETest1() {
        double expected = 16.67;
        double actual = NeuralNetwork.meanSquareError(m1, m2);
        
        assertEquals(expected, actual);
    }

    /**
     * zero case
     */
    @Test
    public void MSETest2() {
        double expected = 259.98;
        double actual = NeuralNetwork.meanSquareError(m1, zeros);

        assertEquals(expected, actual);
    }

    /**
     * Squaring gets rid of negatives.
     */
    @Test
    public void MSETest3() {
        double expected = 259.98;
        double actual = NeuralNetwork.meanSquareError(zeros, m1);

        assertEquals(expected, actual);
    }
    
    /**
     * Bad prediction.
     */
    @Test
    public void crossEntropyTest1() {
        double[][] one = {
                {1}
        };
        Matrix yHat = new Matrix(one);

        double[][] two = {
                {0}
        };
        Matrix y = new Matrix(two);
        
        double expected = -(Math.log(1 - 0));
        double actual = NeuralNetwork.crossEntropyCost(yHat, y);
        
    }

    /**
     * Correct prediction
     */
    @Test
    public void crossEntropyTest2() {
        double[][] one = {
                {1}
        };
        Matrix yHat = new Matrix(one);

        double[][] two = {
                {1}
        };
        Matrix y = new Matrix(two);
        
        double expected = -(Math.log(1));
        double actual = NeuralNetwork.crossEntropyCost(yHat, y);
    }
    
}