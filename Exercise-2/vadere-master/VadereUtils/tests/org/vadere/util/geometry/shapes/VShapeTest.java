package org.vadere.util.geometry.shapes;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.TimeUnit;

import static  org.junit.jupiter.api.Assertions.*;

public class VShapeTest {

	@Test
	public void testIntersectShapesReturnsTrueWithOverlappingShapes() {
		VShape a = new VRectangle(0, 0, 1, 1);
		VShape b = new VRectangle(0, 0, 1, 1);

		assertTrue(a.intersects(b));
	}

	@Timeout(value = 1000,unit = TimeUnit.MILLISECONDS)
	@Test
	public void loopAtEdgeCase1(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(0.005096532915902063, 118.69364126200188), 0.2);
		boolean contains = rec.containsShape(circle);
		assertFalse(contains);
	}

	@Timeout(value = 1000,unit = TimeUnit.MILLISECONDS)
	@Test
	public void loopAtEdgeCase2(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(0.008096532915902063, 118.69364126200188), 0.2);
		boolean contains = rec.containsShape(circle);
		assertFalse(contains);
	}

	@Disabled
	@Timeout(value = 1000,unit = TimeUnit.MILLISECONDS)
	@Test
	/**
	 * Error at magic number. With given Shapes in this test the AWT Area equals comparision
	 * will start loop forever. Small changes (0.005 --> 0.008) will fix this.
	 * Problem occurs if the contains check involves circles because they get transformed
	 * to awt Paths for the comparision (bezier curve) which is probably the reason.
	 * Current fix introduces a new contains(VCircle otherShape) overload in which
	 * specialized check should take place. Currently only VRectangle implement this.
	 * Other shapes will still use the numerical unstable version.
	 */
	public void loopAtEdgeCase_asVPolygon1(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(0.005096532915902063, 118.69364126200188), 0.2);
		VPolygon poly = new VPolygon(rec);
		boolean contains = poly.contains(circle);
		assertFalse(contains);
	}

	@Timeout(value = 1000,unit = TimeUnit.MILLISECONDS)
	@Test
	public void loopAtEdgeCase_asVPolygon2(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(0.008096532915902063, 118.69364126200188), 0.2);
		VPolygon poly = new VPolygon(rec);
		boolean contains = poly.contains(circle);
		assertFalse(contains);
	}

	@Timeout(value = 1000,unit = TimeUnit.MILLISECONDS)
	@Test()
	public void loopAtEdgeCase_asVRectangle(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(0.005096532915902063, 118.69364126200188), 0.2);
		boolean contains = rec.contains(circle);
		assertFalse(contains);
	}

	@Test
	public void containsShapeFalse(){
		VRectangle rec = new VRectangle(0.0, 114.0, 44.0, 5.0);
		VCircle circle = new VCircle(new VPoint(300.0, 300.0), 0.2);
		boolean contains = rec.containsShape(circle);
		assertFalse(contains);
	}

	@Test
	public void testIntersectShapesReturnsFalseWithNonOverlappingShapes() {
		VShape a = new VRectangle(0, 0, 1, 1);
		VShape b = new VRectangle(1.1, 0, 1, 1);

		assertFalse(a.intersects(b));
	}

	@Test
	public void testIntersectShapesReturnsTrueWithOverlappingCircles() {
		VShape a = new VRectangle(0, 0, 1, 1);
		VShape b = new VCircle(new VPoint(-1, -1), 5.0);

		assertTrue(a.intersects(b));
	}

	@Test
	public void testOverlapWithCircles() {
		VShape a = new VCircle(new VPoint(503.9265351385102, 506.9174145081969), 0.195);
		VShape b = new VCircle(new VPoint(504.19098333791044, 506.8493305279853), 0.195);

		assertTrue(a.intersects(b));
	}


}