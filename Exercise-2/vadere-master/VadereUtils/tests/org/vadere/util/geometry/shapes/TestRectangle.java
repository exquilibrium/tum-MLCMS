package org.vadere.util.geometry.shapes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.util.geometry.shapes.VRectangle;

import static  org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Benedikt Zoennchen
 */
public class TestRectangle {

	private VRectangle rect1;
	private VRectangle rect2;

	@BeforeEach
	public void setUp() {
		rect1 = new VRectangle(1.0, 2.0, 10.123, 22.3123);
		rect2 = new VRectangle(1.0, 2.0, 10.123, 22.3123);
	}

	@Test
	public void testEquals() {
		assertEquals(rect1, rect2, "equals() does not work properly.");
	}
}
