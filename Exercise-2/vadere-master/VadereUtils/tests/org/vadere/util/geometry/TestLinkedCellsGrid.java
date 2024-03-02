package org.vadere.util.geometry;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.util.geometry.shapes.VPoint;
import org.vadere.util.logging.Logger;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import static  org.junit.jupiter.api.Assertions.assertEquals;
import static  org.junit.jupiter.api.Assertions.assertFalse;
import static  org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Thorough test of the {@link LinkedCellsGrid}.
 * 
 * 
 */
public class TestLinkedCellsGrid {
	private static Logger logger = Logger
			.getLogger(TestLinkedCellsGrid.class);

	private class NotComparableObject implements PointPositioned {
		public int value;
		public VPoint coord;

		public NotComparableObject(int value, VPoint coord) {
			this.value = value;
			this.coord = coord;
		}

		@Override
		public VPoint getPosition() {
			return coord;
		}
	}

	private static class CoordinatedInteger implements PointPositioned {
		public final Integer number;
		public final VPoint coordinate;

		public CoordinatedInteger(Integer number, VPoint coordinate) {
			this.number = number;
			this.coordinate = coordinate;
		}


		@Override
		public VPoint getPosition() {
			return coordinate;
		}
	}

	private static final double left = 0;
	private static final double top = 0;
	private static final double width = 100;
	private static final double height = 100;
	private static final double sideLength = 1;

	VPoint pos1 = new VPoint(0, 0);
	VPoint pos2 = new VPoint(10, 10);
	VPoint pos3 = new VPoint(50, 10);

	int int1 = 1;
	int int2 = 2;
	int int3 = 3;
	int int4 = 4;

	NotComparableObject obj1 = new NotComparableObject(1, pos1);
	NotComparableObject obj2 = new NotComparableObject(2, pos2);
	NotComparableObject obj3 = new NotComparableObject(3, pos3);
	NotComparableObject obj4 = new NotComparableObject(4, pos3);

	/** linked cells grid with comparable objects */
	private static LinkedCellsGrid<CoordinatedInteger> linkedCellsInteger;
	/** linked cells grid with non comparable objects */
	private static LinkedCellsGrid<NotComparableObject> linkedCellsObject;

	/**
	 * Initializes the linked cells grids so that each test gets a clean object.
	 * 
	 * @throws java.lang.Exception
	 */
	@BeforeEach
	public void setUp() throws Exception {
		linkedCellsInteger = new LinkedCellsGrid<CoordinatedInteger>(left, top, width,
				height, sideLength);
		linkedCellsObject = new LinkedCellsGrid<NotComparableObject>(left, top,
				width, height, sideLength);
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#addObject(PointPositioned)}
	 * . adds several integer objects and tries to retrieve them via
	 * {@link LinkedCellsGrid#getObjects(VPoint, double)}
	 * .
	 */
	@Test
	public void testAddObject() {
		CoordinatedInteger coordinatedInteger1 = new CoordinatedInteger(int1, pos1);
		CoordinatedInteger coordinatedInteger2 = new CoordinatedInteger(int2, pos2);
		CoordinatedInteger coordinatedInteger3 = new CoordinatedInteger(int3, pos3);
		linkedCellsInteger.addObject(coordinatedInteger1);
		linkedCellsInteger.addObject(coordinatedInteger2);
		linkedCellsInteger.addObject(coordinatedInteger3);

		// the values are chosen so that all points should clearly be inside the
		// ball
		VPoint testpos1 = new VPoint(25, 25);
		double testradius1 = 40;
		List<CoordinatedInteger> objects1 = linkedCellsInteger.getObjects(testpos1, testradius1);

		assertEquals(3, objects1.size(),
				"the grid did not add the correct number of objects.");
		assertTrue(objects1.contains(coordinatedInteger1),
				"the first object was not added correctly");
		assertTrue(objects1.contains(coordinatedInteger2),
				"the second object was not added correctly");
		assertTrue(objects1.contains(coordinatedInteger3),
				"the third object was not added correctly");

		// add a new object at exactly the same position as before. should NOT
		// be added
		linkedCellsInteger.addObject(new CoordinatedInteger(int4, pos3));

		List<CoordinatedInteger> objects2 = linkedCellsInteger.getObjects(testpos1, testradius1);
		assertEquals(4, objects2.size(),
				"the grid did not add the object but should have.");

		// add exactly the same object at exactly the same position as before.
		// should also be added
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));

		List<CoordinatedInteger> objects3 = linkedCellsInteger.getObjects(testpos1, testradius1);
		assertEquals(5, objects3.size(),
				"the grid did add the object but should not have.");
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#addObject(PointPositioned)}
	 * . adds several non comparable objects and tries to retrieve them via
	 * {@link LinkedCellsGrid#getObjects(VPoint, double)}
	 * .
	 */
	@Test
	public void testAddNonComparableObject() {
		linkedCellsObject.addObject(obj1);
		linkedCellsObject.addObject(obj2);
		linkedCellsObject.addObject(obj3);

		// the values are chosen so that all points should clearly be inside the
		// ball
		VPoint testpos1 = new VPoint(25, 25);
		double testradius1 = 40;
		List<NotComparableObject> objects1 = linkedCellsObject.getObjects(
				testpos1, testradius1);

		assertEquals(3, objects1.size(),
				"the grid did not add the correct number of objects.");
		assertTrue(objects1.contains(obj1),
				"the first object was not added correctly");
		assertTrue(objects1.contains(obj2),
				"the second object was not added correctly");
		assertTrue(objects1.contains(obj3),
				"the third object was not added correctly");

		// add a new object at exactly the same position as before. should be
		// added
		linkedCellsObject.addObject(obj4);

		List<NotComparableObject> objects2 = linkedCellsObject.getObjects(
				testpos1, testradius1);
		assertEquals(4, objects2.size(),
				"the grid did not add the object but should have.");

		// add exactly the same object at exactly the same position as before.
		// should also be added
		linkedCellsObject.addObject(obj1);

		List<NotComparableObject> objects3 = linkedCellsObject.getObjects(
				testpos1, testradius1);
		assertEquals(5, objects3.size(),
				"the grid did add the object but should not have.");
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#addObject(PointPositioned)}
	 * . adds several points and tests the number of objects stored in the
	 * linked cells grid after adding three comparable objects.
	 */
	@Test
	public void testAddObjectSize() {
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));
		linkedCellsInteger.addObject(new CoordinatedInteger(int2, pos2));
		linkedCellsInteger.addObject(new CoordinatedInteger(int3, pos3));

		assertEquals(3, linkedCellsInteger.size(),
				"size of linkedCellsDouble is wrong");

		// add exactly the same object at exactly the same position as before.
		// should still be added
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos3));

		assertEquals(4, linkedCellsInteger.size(),
				"size of linkedCellsDouble is wrong");
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#getObjects(VPoint, double)}
	 * . Adds objects and tries to retrieve them via getObjects.
	 */
	@Test
	public void testGetObjects() {
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));
		linkedCellsInteger.addObject(new CoordinatedInteger(int2, pos2));
		linkedCellsInteger.addObject(new CoordinatedInteger(int3, pos3));

		// this should only return the object at pos1
		List<CoordinatedInteger> objects = linkedCellsInteger.getObjects(pos1,
				pos1.distance(pos2) - 1);
		assertEquals(
				1, objects.size(),
				"getObjects did not return the correct number of objects.");
		assertEquals(int1, (int) objects.get(0).number,
				"getObjects did not return the correct object.");

		// this should return the object at pos1 and pos2
		List<CoordinatedInteger> objects2 = linkedCellsInteger.getObjects(pos1,
				pos1.distance(pos2) + 1);
		assertEquals(
				2, objects2.size(),
				"getObjects did not return the correct number of objects.");
		assertEquals(int1, (int) objects2.get(0).number,
				"getObjects did not return the correct object.");
		assertEquals(int2, (int) objects2.get(1).number,
				"getObjects did not return the correct object.");
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#removeObject(PointPositioned)}
	 * objects and tries to remove one.
	 */
	@Test
	public void testRemoveObject() {
		CoordinatedInteger coordinatedInteger = new CoordinatedInteger(int2, pos2);
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));
		linkedCellsInteger.addObject(coordinatedInteger);

		// remove the second object
		linkedCellsInteger.removeObject(coordinatedInteger);
		assertEquals(1, linkedCellsInteger.size(),
				"remove object did not remove the object.");
		assertEquals(int1, (int) linkedCellsInteger.iterator().next().number,
				"remove object did not remove the last object.");
	}

	/**
	 * Test method for {@link org.vadere.util.geometry.LinkedCellsGrid#clear()}. Adds two
	 * objects, clears the grid and checks if it is really empty.
	 */
	@Test
	public void testClear() {
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));
		linkedCellsInteger.addObject(new CoordinatedInteger(int2, pos2));

		linkedCellsInteger.clear();

		assertEquals(0, linkedCellsInteger.size(), "the grid is not empty.");

		// try to retrieve the objects, should return an empty list
		List<CoordinatedInteger> objects = linkedCellsInteger.getObjects(pos1,
				pos1.distance(pos2) + 1);
		assertEquals(0, objects.size(),
				"there are still objects present in the grid.");
	}

	/**
	 * Test method for {@link org.vadere.util.geometry.LinkedCellsGrid#iterator()}. Adds three
	 * objects and iterates over them.
	 */
	@Test
	public void testIterator() {
		linkedCellsInteger.addObject(new CoordinatedInteger(int1, pos1));
		linkedCellsInteger.addObject(new CoordinatedInteger(int2, pos2));
		linkedCellsInteger.addObject(new CoordinatedInteger(int3, pos3));

		List<Integer> objects = new LinkedList<Integer>();

		Iterator<CoordinatedInteger> objectIterator = linkedCellsInteger.iterator();
		assertEquals(true, objectIterator.hasNext(),
				"the iterator should find the first element");
		objects.add(objectIterator.next().number);
		assertEquals(true, objectIterator.hasNext(),
				"the iterator should find the second element");
		objects.add(objectIterator.next().number);
		assertEquals(true, objectIterator.hasNext(),
				"the iterator should find the third element");
		objects.add(objectIterator.next().number);
		assertEquals(
				false,
				objectIterator.hasNext(), "the iterator should not find any elements after the third element");

		// check if the correct objects were returned
		assertTrue(objects.contains(int1),
				"the iterator did not return object 1");
		assertTrue(objects.contains(int2),
				"the iterator did not return object 2");
		assertTrue(objects.contains(int3),
				"the iterator did not return object 3");
	}

	/**
	 * Test method for {@link org.vadere.util.geometry.LinkedCellsGrid#size()}. Adds three
	 * objects, removes one object, clears the grid and checks the size before
	 * and after each operation.
	 */
	@Test
	public void testSize() {
		assertEquals(0, linkedCellsInteger.size(), "grid is not empty.");
		CoordinatedInteger coordinatedInteger = new CoordinatedInteger(int1, pos1);
		linkedCellsInteger.addObject(coordinatedInteger);
		linkedCellsInteger.addObject(new CoordinatedInteger(int2, pos2));
		linkedCellsInteger.addObject(new CoordinatedInteger(int3, pos3));

		assertEquals(3, linkedCellsInteger.size(),
				"grid does not contain the correct number of objects.");

		linkedCellsInteger.removeObject(coordinatedInteger);

		assertEquals(2, linkedCellsInteger.size(),
				"grid does not contain the correct number of objects.");

		linkedCellsInteger.clear();

		assertEquals(0, linkedCellsInteger.size(),
				"grid is not empty after clear.");
	}

	/**
	 * Test method for
	 * {@link org.vadere.util.geometry.LinkedCellsGrid#contains(PointPositioned)}. Checks
	 * contains on an empty grid as well as with three added objects.
	 */
	@Test
	public void testContainsT() {
		CoordinatedInteger coordinatedInteger1 = new CoordinatedInteger(int1, pos1);
		CoordinatedInteger coordinatedInteger2 = new CoordinatedInteger(int2, pos2);
		CoordinatedInteger coordinatedInteger3 = new CoordinatedInteger(int3, pos3);
		CoordinatedInteger coordinatedInteger4 = new CoordinatedInteger(int4, pos3);
		assertFalse(linkedCellsInteger.contains(coordinatedInteger1),
				"grid did contain object 1 even it was empty.");



		linkedCellsInteger.addObject(coordinatedInteger1);
		linkedCellsInteger.addObject(coordinatedInteger2);
		linkedCellsInteger.addObject(coordinatedInteger3);

		assertTrue(linkedCellsInteger.contains(coordinatedInteger1),
				"grid did not contain object 1.");
		assertTrue(linkedCellsInteger.contains(coordinatedInteger2),
				"grid did not contain object 1.");
		assertTrue(linkedCellsInteger.contains(coordinatedInteger3),
				"grid did not contain object 1.");
		assertFalse(linkedCellsInteger.contains(coordinatedInteger4),
				"grid did contain object 4.");
	}

	/**
	 * Test method for the complexity of
	 * {@link LinkedCellsGrid#getObjects(VPoint, double)}. Should be O(1).
	 */
	@Test
	public void testGetObjectsCompexity() {
		// throw a lot of objects in the grid, equally spaced.
		int[] objCounts = new int[] {10, 100, 1000, 10000, 100000};

		List<Long> times = new LinkedList<Long>();

		for (int count : objCounts) {
			int numberOfObjects = count;

			// create a grid that holds at max one object per cell
			double sideLength = width / Math.sqrt(count);
			linkedCellsInteger = new LinkedCellsGrid<>(left, top, width,
					height, sideLength);
			linkedCellsInteger.clear();
			fillGrid(linkedCellsInteger, numberOfObjects);

			// access the grid a lot of times, compute the time needed
			int numberOfSearches = (int) 1e6;
			VPoint searchPos = new VPoint(width / 2, height / 2);
			double radius = sideLength * 2;
			long startTime = System.currentTimeMillis();

			for (int search = 0; search < numberOfSearches; search++) {
				linkedCellsInteger.getObjects(searchPos, radius);
			}

			long totalTime = System.currentTimeMillis() - startTime;

			times.add(totalTime);
			logger.debug(String.format(
					"searching %d objects %d times took %d ms.", count,
					numberOfSearches, totalTime));
		}

		// test whether the times are similar
		double mean = mean(times);
		logger.debug(String.format("mean of all search times: %.2f ms", mean));
		for (int time = 1; time < times.size(); time++) {
			assertTrue(times.get(time) < mean * 3,
					"getObjects took too much time.");
		}
	}

	private double mean(List<Long> times) {
		double mean = 0.0;
		for (Long time : times) {
			mean += time;
		}
		return mean / times.size();
	}

	/**
	 * Fills an integer grid with a given number of objects.
	 * 
	 * @param linkedCellsGrid
	 * @param numberOfObjects
	 */
	private void fillGrid(LinkedCellsGrid<CoordinatedInteger> linkedCellsGrid,
			int numberOfObjects) {
		int size = (int) Math.sqrt(numberOfObjects);
		for (int row = 0; row < size; row++) {
			for (int col = 0; col < size; col++) {
				int obj = row * size + col;
				VPoint pos = new VPoint(row / (double) size * height, col
						/ (double) size * width);
				linkedCellsGrid.addObject(new CoordinatedInteger(obj, pos));
			}
		}
	}

}
