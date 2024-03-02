package org.vadere.state.scenario;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.state.attributes.scenario.AttributesAerosolCloud;
import org.vadere.util.geometry.shapes.VCircle;
import org.vadere.util.geometry.shapes.VPoint;
import org.vadere.util.geometry.shapes.VShape;

import java.util.ArrayList;
import java.util.Arrays;

import static org.vadere.state.attributes.Attributes.ID_NOT_SET;


public class AerosolCloudTest {
    public static double ALLOWED_DOUBLE_TOLERANCE = 10e-3;

    protected AerosolCloud aerosolCloud = new AerosolCloud();
    private AerosolCloud aerosolCloudCirc;


    @BeforeEach
    public void setUp() {
        // common attributes
        int id = ID_NOT_SET;
        double radius = 1;
        VPoint center = new VPoint(10, 10);
        double creationTime = 0;
        double initialPathogenLoad = 10e4;

        AttributesAerosolCloud attributesCirc = new AttributesAerosolCloud(id, radius, center, creationTime, initialPathogenLoad);
        aerosolCloudCirc = new AerosolCloud(attributesCirc);
    }


    @Test
    public void testGetId() {
        int expectedId = ID_NOT_SET;
        assertEquals(expectedId, aerosolCloud.getId());
    }


    @Test
    public void testSetShape() {
        VCircle newShape = new VCircle(new VPoint(10, 10), 2);

        aerosolCloud.setShape(newShape);
        assertEquals(newShape, aerosolCloud.getShape());
    }


    @Test
    public void testTestClone() {
        int id = 1;
        double radius = 3.0;
        VPoint center = new VPoint(2, 2);
        VCircle shape = new VCircle(center, radius);

        VCircle newShape = new VCircle(new VPoint(3, 3), 4);
        double creationTime = 999.9;
        double pathogenLoad = 10e9;
        double lifeTime = 60*60*3;

        AerosolCloud aerosolCloudOriginal = new AerosolCloud(new AttributesAerosolCloud(id, radius, center, creationTime, pathogenLoad));
        AerosolCloud aerosolCloudClone = aerosolCloudOriginal.clone();

        assertEquals(aerosolCloudOriginal.getId(), aerosolCloudClone.getId());
        assertEquals(aerosolCloudOriginal.getShape(), aerosolCloudClone.getShape());
        assertEquals(aerosolCloudOriginal.getCreationTime(), aerosolCloudClone.getCreationTime(), ALLOWED_DOUBLE_TOLERANCE);

        // check that original is not affected by setters/changes to the clone
        aerosolCloudClone.setId(2);
        aerosolCloudClone.setShape(newShape);
        aerosolCloudClone.setCreationTime(0);

        assertEquals(id, aerosolCloudOriginal.getId());
        assertEquals(shape, aerosolCloudOriginal.getShape());
        assertEquals(creationTime, aerosolCloudOriginal.getCreationTime(), ALLOWED_DOUBLE_TOLERANCE);
    }

    @Test
    public void throwIfIncreaseShapeChangesCenterOfCircularAerosolCloud() {
        VPoint center = aerosolCloudCirc.getCenter();
        aerosolCloudCirc.increaseShape(0.2);
        assertEquals(center, aerosolCloudCirc.getCenter());
    }

}