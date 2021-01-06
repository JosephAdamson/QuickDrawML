package utils;
 
import java.io.Serializable;
import java.util.Random;
import java.util.function.Function;

/**
 * <pre>
 * Minimal matrix library supplying basic operations to be used in 
 * conjunction with my NeuralNetwork.Network class. Contains overloaded methods for 
 * each operation with different parameters (scalar, elementwise etc.) 
 * The indices of a matrix object's elements are zero indexed as it uses
 * a 2d array to internally store its data.
 *
 * E.g. a 3 x 4 matrix would be represented as a traditional 2d array:
 * [[0, 1, 0, 1]
 *  [1, 1, 0, 0]
 *  [0, 0, 1, 0]]
 * </pre>
 *
 * @author Joseph Adamson
 * @version 07.07.2020
 */
public class Matrix implements Serializable {

    /**
     * The dimensions of a matrix object.
     */
    private final int rows, cols;

    /**
     * Matrix elements are stored in a 2d array.
     */
    private final double[][] data;

    /**
     * Tolerance used for determining the equality of two
     * doubles in .equals() method.
     */
    public static final double TOLERANCE = 0.00000000001;

    /**
     * Constructs a matrix from a 2d array.
     *
     * @param inputData 2d array to copy.
     */
    public Matrix(double[][] inputData) {
        this.rows = inputData.length;
        this.cols = inputData[0].length;
        
        // Check that cols of input data are equal.
        for (double[] row : inputData) {
            if (row.length != this.cols) {
                throw new IllegalArgumentException("Input data is malformed");
            }
        }
        this.data = new double[rows][cols];

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.data[i][j] = inputData[i][j];
            }
        }
    }

    /**
     * Constructs a matrix object from an array.
     *
     * @param inputData: an array object
     */
    public Matrix(double[] inputData) {
        this.rows = 1;
        this.cols = inputData.length;
        this.data = new double[rows][cols];

        for (int i = 0; i < inputData.length; i++) {
            this.data[0][i] = inputData[i];
        }
    }

    /**
     * Constructs a new matrix of the specified dimensions, where all 
     * elements are zero.
     *
     * @param rows number of specified rows for the matrix.
     * @param cols number of specified columns for the matrix.
     */
    public Matrix(int rows, int cols) {
        if (rows < 0 || cols < 0 ) {
            throw new NegativeArraySizeException("A matrix cannot have" +
                    " negative dimensions.");
        } else {
            this.rows = rows;
            this.cols = cols;
            this.data = new double[rows][cols];
        }
    }

    /**
     * Constructs a copy of parameter matrix.
     * 
     * @param toClone a matrix object
     */
    public Matrix(Matrix toClone) {
        this.rows = toClone.rows;
        this.cols = toClone.cols;
        this.data = toClone.data;
    }

    /**
     * @return number of rows. 
     */
    public int getRows(){
        return this.rows;
    }

    /**
     * @return number of columns.
     */
    public int getCols() {
        return this.cols;
    }

    /**
     * @return internal data of the matrix.
     */
    public double[][] getData() {
        return data;
    }
    
    /**
     * Returns a new matrix (rows x cols) of randomly generated values.
     *
     * @param rows number of specified rows for the matrix.
     * @param cols number of specified columns for the matrix.
     * @return a new rows x cols matrix.
     */
    public static Matrix randomize(int rows, int cols) {
        if (rows < 0 || cols < 0 ) {
            throw new NegativeArraySizeException("A matrix cannot have" +
                    " negative dimensions.");
        } else {
            Random rand = new Random();

            Matrix product = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {

                    // Values between initialised between -1 and 1.
                    product.data[i][j] = (1 - (-1)) * rand.nextDouble() + (-1);
                }
            }
            return product;
        }
    }

    /**
     * 'Reshape' the dimensions of a given matrix by creating a new
     * matrix out of A's updated internal data.
     * 
     * @param A a matrix object.
     * @param rows the number of rows for the new reshaped matrix 
     * @param cols the number of cols for the new reshaped matrix
     * @return a reshaped matrix with the dimensions rows x cols
     */
    public static Matrix reshape(Matrix A, int rows, int cols) {
        int totalElements = A.rows * A.cols;
        
        // check to see if the dimensions of the result matrix are feasible
        // given A's current internal data.
        if (totalElements != rows * cols || totalElements % rows != 0) {
            throw new IllegalArgumentException("Cannot reshape matrix with current dimensions");
        } else if (rows < 0 || cols < 0) {
            throw new NegativeArraySizeException("A matrix cannot have" +
                    " negative dimensions.");
        } else {
            final Matrix result = new Matrix(rows, cols);

            int newRow = 0;
            int newCol = 0;
            for (int i = 0; i < A.data.length; i++) {
                for (int j = 0; j < A.data[i].length; j++) {
                    result.data[newRow][newCol] = A.data[i][j];
                    newCol++;

                    if (newCol == cols) {
                        newCol = 0;
                        newRow++;
                    }
                }
            }
            return result;
        }
    }

    /**
     * Elementwise addition of matrices A and B.
     *
     * @param A a matrix object.
     * @param B a matrix object.
     * @return the sum of matrix A and B.
     */
    public static Matrix add(Matrix A, Matrix B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new IllegalArgumentException("Parameter matrices do not have " +
                    "corresponding dimensions for add operation.");
        } else {
            Matrix result = new Matrix(A.rows, A.cols);
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < A.cols; j++) {
                    result.data[i][j] = A.data[i][j] + B.data[i][j];
                }
            }
            return result;
        }
    }

    /**
     * Elementwise addition with a scalar number.
     *
     * @param A A matrix
     * @param x the scalar you want to add to each element in A
     * @return a matrix where each element is A.data[i][j] + x
     */
    public static Matrix add(Matrix A, double x) {
        Matrix result = new Matrix(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                result.data[i][j] = A.data[i][j] += x;
            }
        }
        return result;
    }

    /**
     * Elementwise subtraction of matrices A and B.
     *
     * @param A a matrix object.
     * @param B a matrix object.
     * @return the sum of matrix A and B.
     */
    public static Matrix subtract(Matrix A, Matrix B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new IllegalArgumentException("Parameter matrices do not have " +
                    "corresponding dimensions for subtract operation.");
        } else {
            Matrix result = new Matrix(A.rows, A.cols);
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < A.cols; j++) {
                    result.data[i][j] = A.data[i][j] - B.data[i][j];
                }
            }
            return result;
        }
    }

    /**
     * Elementwise subtraction with A scalar number.
     *
     * @param x the scalar you want to subtract from each element in A.
     * @return A matrix where each element is A.data[i][j] - x
     */
    public static Matrix subtract(Matrix A, double x) {
        Matrix result = new Matrix(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                result.data[i][j] = A.data[i][j]  -= x;
            }
        }
        return result;
    }

    /**
     * Dot product multiplication of A and B. 
     *
     * @param A a matrix object.
     * @param B a matrix object.
     * @return a new matrix where each element is the dot product of row 
     * i in matrix A and col i in matrix B. 
     */
    public static Matrix dotProduct(Matrix A, Matrix B) {

        // The number of columns of A must be equal to the number
        // of rows in B.
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("Parameter matrices do not have " +
                    "corresponding dimensions for multiplication.");
        } else {
            Matrix result = new Matrix(A.rows, B.cols);
            for (int i = 0; i < result.rows; i++) {
                for (int j = 0; j < result.cols; j++) {

                    double elementSum = 0;
                    
                    // Each element in a given row of matrix A is multiplied elementwise
                    // with the corresponding element in a given column in matrix B.
                    // The result for [i][j] in our new array is the dot product of all
                    // the row spots of the column in A and all the column spots in B.
                    for (int k = 0; k < A.cols; k++) {
                        elementSum += (A.data[i][k] * B.data[k][j]);
                    }
                    result.data[i][j] = elementSum;
                }
            }
            return result;
        }
    }

    /**
     * Hadamard product multiplication of matrix A and B 
     * (elementwise multiplication). 
     *
     * @param A a matrix object.
     * @param B a matrix object.
     * @return the Hadamard product of A and B.
     */
    public static Matrix hadamardProduct(Matrix A, Matrix B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new IllegalArgumentException("Parameter matrix does not have " +
                    "corresponding dimensions for multiplication.");
        } else {
            Matrix result = new Matrix(A.rows, B.cols);
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < A.cols; j++) {
                    result.data[i][j] = A.data[i][j] *= B.data[i][j];
                }
            }
            return result;
        }
    }

    /**
     * Elementwise multiplication with a scalar number.
     *
     * @param A a matrix object.
     * @param x: the scalar number you want to multiply each element
     * in A by.
     */
    public static Matrix multiply(Matrix A, double x) {
        Matrix result = new Matrix(A.rows, A.cols);
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                result.data[i][j] = A.data[i][j] *= x;
            }
        }
        return result;
    }

    /**
     * Method transposes matrix A; each column is rotated 90 degrees
     * (becoming a row in the result matrix) and the elements are
     * reversed.
     *
     * @param A the matrix to be transposed.
     * @return A transposed.
     */
    public static Matrix transpose(Matrix A) {
        Matrix result = new Matrix(A.cols, A.rows);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                result.data[i][j] = A.data[j][i];
            }
        }
        return result;
    }

    /**
     * Returns the index of the row containing the greatest value
     * in a given matrix.
     * 
     * @param A a matrix object
     * @return an integer corresponding to a row index.
     */
    public static int argMaxRow(Matrix A) {
        int maxRow = 0;
        double maxVal = A.data[0][0];
        
        for (int i = 0; i < A.rows; i++) {
           for (int j = 0; j < A.cols; j++) {
               
               if (A.data[i][j] > maxVal) {
                   maxVal = A.data[i][j];
                   maxRow = i;
               }
           }
        }
        return maxRow;
    }

    /**
     * Computes the sum of all elements in matrix A
     * 
     * @param A a matrix object. 
     * @return the sum of all elements in A
     */
    public static double sum(Matrix A) {
        double sum = 0;
        
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                sum += A.data[i][j];
            }
        }
        return sum;
    }

    /**
     * Elementwise mapping of a function to A
     *
     * @param f: an activation function.
     */
    public static Matrix map(Function<Double, Double> f, Matrix A) {
        Matrix result = new Matrix(A.rows, A.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                result.data[i][j] = f.apply(A.data[i][j]);
            }
        }
        return result;
    }

    /**
     * Comparison of matrices A and B.
     * 
     * @param other: matrix for comparison. 
     * @return true if other contains the same data as the calling
     *         matrix.
     */
    public boolean equals(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new RuntimeException("Parameter matrix does not have " +
                    "corresponding dimensions");
        } else {
            for (int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    if (Math.abs(this.data[i][j] - other.data[i][j]) > TOLERANCE) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Renders matrix object in a readable format.
     *
     * @return String representation of a utils.Matrix.
     */
    @Override
    public String toString(){
        StringBuilder result = new StringBuilder();

        result.append("[");
        for (int i = 0; i < data.length; i++){
            if (i == 0) {
                result.append("[");
            } else {
                result.append(" [");
            }
            for (int j = 0; j < data[i].length; j++){
                if (j == data[i].length - 1){
                    if (i == data.length - 1) {
                        result.append(data[i][j]).append("]");
                    } else {
                        result.append(data[i][j]).append("]\n");
                    }
                } else {
                    result.append(data[i][j]).append("\t ");
                }
            }
        }
        result.append("]\n");
        return result.toString();
    }
    
    public static void main(String[] args) {
        Matrix test = new Matrix(3, 6);
        System.out.println(test);
        
    }
}
