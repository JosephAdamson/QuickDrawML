package knearestneighbours;

import java.util.Comparator;

/**
 * Comparator used in conjunction with Arrays.sort() to sort
 * Distance objects by distance.
 */
public class DistanceComp implements Comparator<Distance> {
    public static final double TOLERANCE = 0.0000000001;
    
    @Override
    public int compare(Distance d1, Distance d2) {
        if (Math.abs(d1.getDistance() - d2.getDistance()) < TOLERANCE) {
            return 0;
        } else if (d1.getDistance() > d2.getDistance()) {
            return 1;
        } else {
            return -1;
        }
    }
}
