package utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DataPrep {

    /**
     * The total number of pixels in each 28 x 28 image
     */
    public static final int PIXELS = 784;

    /**
     * Iterates through the raw data folder, takes a slice from each .npy file
     * representing x amount of sample images and combines these slices to create
     * a .dat file used for either training, validating or testing. The output .dat
     * file is a serialized double array. Each row has 785 elements; 784 pixel values
     * and an additional label (an integer 0 - 4).
     * 
     * @param sampleSize  the number of images recruited from the original dataset.
     * @param sampleStart the image number (ordinal) you want to start from in terms 
     *                    of loading bytes into the return array.
     * @throws IOException if method cannot access the data directory.
     */
    public static void packData(int sampleSize, int sampleStart, String filename) throws IOException {
        File rawDirectory = new File(System.getProperty("user.dir") + "/data/raw/");
        File[] fileListing = rawDirectory.listFiles();
        System.out.println(Arrays.toString(fileListing));

        if (fileListing != null) {

            double[][] result = new double[sampleSize * fileListing.length][785];

            for (int f = 0; f < fileListing.length; f++) {
                double[][] sample =
                        prepareRawData(fileListing[f], sampleSize, sampleStart, f);
                System.arraycopy(sample, 0, result, f * sampleSize, sample.length);
            }
            try {
                FileOutputStream fos = new FileOutputStream(filename);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(result);
            } catch (Exception e) {
                System.err.println("Directory no found");
            }
        } else {
            throw new IOException("Directory not found.");
        }
    }
    
    /**
     * Method creates a 2d array of flattened images from a single .npy file
     * (category).
     * 
     * @param file        .npy category file 
     * @param sampleSize  the number of images recruited from the original dataset.
     * @param sampleStart the image number (ordinal) you want to start from in terms 
     *                    of loading bytes into the return array.
     * @param label       a give sample's corresponding label.
     * @return a 2d array where each row contains 785 values; 0 - 784 contains normalized
     * pixel values with the last denoting the label of the image.
     */
    public static double[][] prepareRawData(File file, int sampleSize, int sampleStart, int label) {
        Path path = file.toPath();
        double[][] result = new double[sampleSize][785];
        
        try {
            
            // convert the entire .npy source file into a byte array.
            byte[] raw = Files.readAllBytes(path);
            if (sampleSize < 0 || sampleSize > raw.length / PIXELS) {
                throw new IllegalArgumentException("Sample size incompatible with the" +
                        "provided data");
            } else {
                double[] normalized = new double[(sampleSize + 1) * PIXELS];

                int lowerBound = (sampleStart * PIXELS) + 80;
                int upperBound = lowerBound + (sampleSize * PIXELS) + 80;
                int index = 0;
                for (int i = lowerBound; i < upperBound; i++) {

                    // apply bitwise operation to convert signed bytes into
                    // values between 0 - 255, then normalize these values. 
                    int val = raw[i] & 0xff;
                    normalized[index] = (double) val / 255;
                    index++;
                }
                
                for (int j = 0; j < sampleSize; j++) {
                    int start = j * PIXELS;
                    double[] temp = Arrays.copyOfRange(normalized, start, start + PIXELS);
                    System.arraycopy(temp, 0, result[j], 0, temp.length);
                    result[j][result[j].length -1] = label;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    /**
     * Load a .dat file for use with a model.
     *
     * @param filename a .dat file used to train, validate or test a model.
     * @return the contents of the .dat file represented as a 2d array of type double.
     * @throws IOException if parameter file cannot be found.
     */
    public static double[][] loadData(File filename) throws IOException {
        double[][] result = null;
        try {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            result = (double[][]) ois.readObject();
        } catch (Exception e) {
            System.err.println("File not found");
        }

        if (result == null) {
            throw new IOException("File not found");
        } else {
            return result;
        }
    }

    /**
     * Converts a 2d array of packed data into (feature, label) column matrix 
     * tuples that can easily be processed by the neural network.
     * 
     * @param data a double array of 'packed' data where each row represents a 
     *             flattened image.
     * @return a 2d ArrayList of matrices; each row is a (features, label) pair
     * of vectorized matrices.
     */
    public static ArrayList<ArrayList<Matrix>> vectorize(double[][] data, int outputs) {
        ArrayList<ArrayList<Matrix>> result = new ArrayList<>();

        for (double[] row : data) {
            ArrayList<Matrix> annotation = new ArrayList<>();
            Matrix features =
                    Matrix.reshape(
                            new Matrix(Arrays.copyOfRange(row, 0, row.length - 1)),
                            row.length - 1, 1
                    );
            Matrix label = oneHotEncode(row[row.length - 1], outputs);
            annotation.add(features);
            annotation.add(label);
            result.add(annotation);
        }
        return result;
    }

    /**
     * Partitions data into subsets.
     *
     * @param training a 2d array where each row contains 785 values; 0 - 784 contains 
     * normalized pixel values with the last denoting the label of the image.
     * @param K determines the number of subsets the original data will be split
     * into.
     * @return an arrayList containing K elements.
     */
    public static ArrayList<double[][]> split(double[][] training, int K) {
        if (training.length % K != 0) {
            throw new IllegalArgumentException("Data must be divisible by fold size");
        } else {
            int sampleSize = training.length / K;

            ArrayList<double[][]> folded = new ArrayList<>();
            for (int i = 0; i < K; i++) {
                int start = i * sampleSize;
                double[][] fold = Arrays.copyOfRange(training, start, start + sampleSize);
                folded.add(fold);
            }
            return folded;
        }
    }

    /**
     * Coverts label data into a binary column matrix; each row corresponds
     * to a particular output label. The label represented is indicated with 1.0.
     * 
     * @param label an integer value (indexed 0 - 9 and ordered alphabetically) 
     *              representing a picture category (axe, skull etc.)
     * @return a matrix of binary double values 
     *        
     */
    public static Matrix oneHotEncode(double label, int outputs) {
        if (label > outputs) {
            throw new IllegalArgumentException("numerical label cannot be larger than" +
                    "the provided outputs");
        } else if (label < 0) {
            throw new IllegalArgumentException("label cannot be a negative number"); 
        } else {
            double[] oneHot = new double[outputs];
            oneHot[(int)label] = 1.0;
            return Matrix.reshape(new Matrix(oneHot),oneHot.length, 1);
        }
    }

    /**
     * Shuffles rows in a 2d array.
     *
     * @param data a 2d array. 
     * @return the original array with the rows shuffled.
     */
    public static double[][] shuffleData(double[][] data) {
        List<double[]> toList = Arrays.asList(data);
        Collections.shuffle(toList);
        return toList.toArray(new double[0][0]);
    }

    /**
     * Concatenate two 2 2d arrays of type double.
     * 
     * @param a a 2d array
     * @param b a 2d array
     * @return a 2d array a + b
     */
    public static double[][] append(double[][] a, double[][] b) {
        double[][] result = new double[a.length + b.length][];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
    
    //------Additional methods used to access/view data-------

    /**
     * Once the .dat file has been converted into a 2d array individual samples
     * (individual arrays from the 2d array) can be viewed 
     * @param sample the sample to to viewed (ordinal) from the data
     */
    public static void viewSample(double[] sample, int sampleNum) {
        try {
            BufferedImage image =
                    new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            WritableRaster raster = image.getRaster();
            DataBufferByte buffer = (DataBufferByte) raster.getDataBuffer();
            byte[] b = buffer.getData();

            for (int i = 0; i < PIXELS; i++) {
                int pixel = (int) (sample[i] * 255);
                b[i] = (byte) (255 - pixel);
            }

            File output = new File("output" + sampleNum + ".jpg");
            ImageIO.write(image, "jpg", output);
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) throws IOException {
        /*packData(2800, 0, "train.dat");
        packData(600, 10000, "validation.dat");
        packData(600, 20000, "test.dat");
        */
        double[][] samples = 
                loadData(new File(System.getProperty("user.dir") + "/data/setB/train.dat"));
        
        for (int i = 0; i < 5; i++) {
            viewSample(samples[8400 + i], i); 
        }
        
    }
}
