package utils;

import org.knowm.xchart.*;
import org.knowm.xchart.style.markers.SeriesMarkers;
import java.awt.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * Class provides simple plotting functions to allow output results 
 * from the models to be further analysed.
 * 
 * @author Joseph Adamson.
 */
public class Metrics {

    /**
     * Plots a simple line graph.
     * 
     * @param title title for the graph window
     * @param xAxis title for the x axis
     * @param yAxis title for the y axis
     * @param seriesName title for the series
     * @param yData y coordinates for the graph
     */
    public static void plotCrossValidationResults(String title, String xAxis, String yAxis,
                                                  String seriesName, double[] yData) {

        // Create the chart window
        XYChart chart = new XYChartBuilder()
                .width(1000)
                .height(600)
                .title(title)
                .xAxisTitle(xAxis)
                .yAxisTitle(yAxis)
                .build();

        // x axis information for the k-range (1 - 30) covered by the 
        // kFoldCrossValidation method
        double[] xData = new double[100];
        for (int i = 0; i < 100; i++) {
            xData[i] = i + 1;
        }
        
        chart.getStyler().setYAxisDecimalPattern("0.000");
        XYSeries series = chart.addSeries(seriesName, xData, yData);
        series.setMarker(SeriesMarkers.NONE);
        new SwingWrapper<>(chart).displayChart();
    }

    /**
     * Plot the cost functions for the neural network.
     * 
     * @param epochs: x axis values
     * @param ySeries: a variable number of cost functions to be plotted.
     */
    public static void plotCostFunction(int epochs, double[][] ySeries) {

        // getting points for the x axis.
        double[] xData = new double[epochs];
        for (int i = 0; i < epochs; i++) {
            xData[i] = i;
        }

        ArrayList<XYChart> charts = new ArrayList<>();

        String xAxis = "epochs";
        String yAxis = "cost";

        XYChart chart = new XYChartBuilder()
                .xAxisTitle(xAxis)
                .yAxisTitle(yAxis)
                .width(500)
                .height(300)
                .build();

        XYSeries costA1 = chart.addSeries("Alpha = 1", xData, ySeries[0]);
        XYSeries costA2 = chart.addSeries("Alpha = 0.1" , xData, ySeries[1]);
        XYSeries costA3 = chart.addSeries("Alpha = 0.01", xData, ySeries[2]);
        costA1.setMarker(SeriesMarkers.NONE);
        costA2.setMarker(SeriesMarkers.NONE);
        costA3.setMarker(SeriesMarkers.NONE);
        costA2.setLineColor(Color.GREEN);
        costA2.setLineColor(Color.MAGENTA);
        charts.add(chart);
        new SwingWrapper<>(charts).displayChartMatrix();
    }

    /**
     * Plots the cost and accuracies for each epoch in a round of testing/validation
     * for the network. Results are displayed in a simple graph matrix.
     * 
     * @param epochs The number of passes through the full training data.
     * @param ySeries: double array containing y coordinates for the cost and accuracies.
     */
    public static void plotNetworkResults(int epochs, double[][] ySeries) {

        // getting points for the x axis.
        double[] xData = new double[epochs];
        for (int i = 0; i < epochs; i++) {
            xData[i] = i;
        }

        ArrayList<XYChart> charts = new ArrayList<>();
        
        String xAxis = "epochs";
        String[] yAxis = new String[]{"cost", "accuracy"};
        for (int i = 0; i < ySeries.length; i+=2) {

            // create a separate chart windows to plot both cost and recall for
            // training and validation sets.
            XYChart chart = new XYChartBuilder()
                    .xAxisTitle(xAxis)
                    .yAxisTitle(i == 2 ? yAxis[1] : yAxis[0])
                    .width(500)
                    .height(300)
                    .build();
            
            XYSeries cost = chart.addSeries("training", xData, ySeries[i]);
            XYSeries recall = chart.addSeries("validation", xData, ySeries[i + 1]);
            cost.setMarker(SeriesMarkers.NONE);
            recall.setMarker(SeriesMarkers.NONE);
            recall.setLineColor(Color.ORANGE);
           
            charts.add(chart);
        }
        new SwingWrapper<>(charts).displayChartMatrix();
    }

    public static double modelAccuracy(int[][] results) {
        int[] testLabels = results[0];
        int[] predictions = results[1];

        double accuracy = 0.0;
        for (int i = 0; i < testLabels.length; i++) {
            if (testLabels[i] == predictions[i]) {
                accuracy++;
            }
        }
        return (accuracy / testLabels.length);
    }

    /**
     * <pre>
     * Prints a matrix that summarizes the prediction results on
     * a set of test data E.g.
     *
     *             class 1    class 2
     *             predicted  predicted
     *   class 1 [   TP         FN   ]
     *   actual
     *
     *   class 2 [   FP         TN   ]
     *   actual
     * </pre>
     * 
     * @param results a 2d array; first row containing actual labels for the test
     *                data the second, corresponding predictions made by a classifier.
     * @return a 2d array representation of the confusion matrix; 
     *         purely for testing purposes. 
     */
    public static double[][] confusionMatrix(int[][] results) {
        
        int[] testLabels = results[0];
        int[] predictions = results[1];

        // get the number of classes for the dimensions of the 
        // confusion matrix, these will always be equal.
        Set<Integer> classCount = new HashSet<>();
        for (int label : testLabels) {
            classCount.add(label);
        }
        int classes = classCount.size();

        double[][] confusion = new double[classes][classes];

        for (int i = 0; i < testLabels.length; i++) {
            confusion[testLabels[i]][predictions[i]]++;
        }

            double confusionSum = Matrix.sum(new Matrix(confusion));

            // Get true positives.
            double[] TP = new double[classes];
            for (int i = 0; i < confusion.length; i++) {
                TP[i] += confusion[i][i];
            }

            // obtaining the row and column sums of the confusion matrix
            // is necessary to be able to compute, true negatives, false positives, 
            // false negatives and overall accuracy.
            double[] colSums = new double[classes];
            double[] rowSums = new double[classes];
            for (int i = 0; i < confusion.length; i++) {
                double colSum = 0.0;
                double rowSum = 0.0;
                for (int j = 0; j < confusion.length; j++) {
                    colSum += confusion[j][i];
                    rowSum += confusion[i][j];
                }
                colSums[i] = colSum;
                rowSums[i] = rowSum;
            }
            
            double[] FP = new double[classes];
            double[] FN = new double[classes];
            double[] TN = new double[classes];
            for (int i = 0; i < classes; i++) {
                FP[i] = colSums[i] - TP[i];
                FN[i] = rowSums[i] - TP[i];
                TN[i] = confusionSum - (rowSums[i] - confusion[i][i]) - colSums[i];
            }

            double[] precision = new double[classes];
            double[] recall = new double[classes];

            // conditions to catch NaN values (caused by dividing by zero)
            for (int i = 0; i < classes; i++) {
                if (TP[i] == 0 && (TP[i] + FP[i]) == 0) {
                    precision[i] = 0.0;
                } else if (TP[i] == 0 && (TP[i] + FN[i]) == 0) {
                    recall[i] = 0.0;
                } else {
                    precision[i] = TP[i] / (TP[i] + FP[i]);
                    recall[i] = TP[i] / (TP[i] + FN[i]);
                }
            }
            
            // Total model accuracy.
            double acc = Matrix.sum(new Matrix(TP)) / predictions.length; 
            
            // sanity check 
            /*for (int i = 0; i < classes; i++) {
                System.out.println((TP[i] + TN[i] + FP[i] + FN[i] == testLabels.length));
            }*/
         
            System.out.println("\n" + new Matrix(confusion));
            System.out.println("============Report============");
            System.out.printf("%18s%10s%n", "Precision", "Recall");

            for (int i = 0; i < classes; i++) {
                System.out.printf("class %d  %9.4f %9.4f %n", i,
                        precision[i], recall[i]);
            }
            System.out.printf("%nModel accuracy: %.4f%n", acc);
            System.out.println("==============================");
            return confusion;
        }
}
