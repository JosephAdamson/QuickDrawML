package knearestneighbours;

/**
 * Wrapper class used to store (distance, label) tuples.
 */
public class Distance {

    /**
     * The respective Euclidean distance between a pair of
     * features (Y, X).
     */
    private double distance;

    /**
     * The label corresponding to feature set X.
     */
    private double label;

    public Distance(double distance, double label) {
        this.distance = distance;
        this.label = label;
    }

    public double getDistance() {
        return distance;
    }

    public double getLabel() {
        return label;
    }
}


