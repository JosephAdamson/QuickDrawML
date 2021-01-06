package knearestneighbours;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class DistanceTest {
    
    private Distance d1;
    private Distance d2;
    private Distance d3;
    private Distance d4;
    private Distance d5;
    
    @BeforeEach
    public void init() {
        d1 = new Distance(505.3, 3);
        d2 = new Distance(30.7, 6);
        d3 = new Distance(7000.5, 2);
        d4 = new Distance(-505.3, 3);
        d5 = new Distance(30.7, 5);
    }
    
    //-----getters-----
    
    @Test
    public void getterTest1() {
        double expected = 505.3;
        double actual = d1.getDistance();
        
        assertEquals(expected, actual);
    }

    /**
     * Trailing zeros
     */
    @Test
    public void getterTest2() {
        double expected = 505.3000000;
        double actual = d1.getDistance();
        
        assertEquals(expected, actual);
    }

    @Test
    public void getterTest3() {
        double expected = 3;
        double actual = d1.getLabel();

        assertEquals(expected, actual);
    }

    //---sorting with comparator----
    
    @Test
    public void comparatorTest1() {
       Distance[] dArray = {d1, d2, d3};
       
       Distance[] expected = {d2, d1, d3};
       Arrays.sort(dArray, new DistanceComp());
       
       for (int i = 0; i < dArray.length; i++) {
           assertEquals(dArray[i].getDistance(), expected[i].getDistance());
       }
    }

    /**
     * Sorting with a negative distance.
     */
    @Test
    public void comparatorTest2() {
        Distance[] dArray = {d1, d2, d3, d4};
        
        Distance[] expected = {d4, d2, d1, d3};
        Arrays.sort(dArray, new DistanceComp());
        
        
        for (int i = 0; i < dArray.length; i++) {
            assertEquals(dArray[i].getDistance(), expected[i].getDistance());
        }
    }

    /**
     * Sorting a singleton
     */
    @Test
    public void comparatorTest3() {
        Distance[] dArray = {d1};

        Distance[] expected = {d1};
        Arrays.sort(dArray, new DistanceComp());


        for (int i = 0; i < dArray.length; i++) {
            assertEquals(dArray[i].getDistance(), expected[i].getDistance());
        }
    }

    /**
     * Sorting with duplicates
     */
    @Test
    public void comparatorTest4() {
        Distance[] dArray = {d1, d2, d3, d4, d5};

        Distance[] expected = {d4, d2, d5, d1, d3};
        Arrays.sort(dArray, new DistanceComp());


        for (int i = 0; i < dArray.length; i++) {
            assertEquals(dArray[i].getDistance(), expected[i].getDistance());
        }
    }
    
    
}