package org.vadere.state.scenario;

import static  org.junit.jupiter.api.Assertions.*;

import java.util.NoSuchElementException;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestDeprecatedPedestrian extends TestPedestrian {

	@Override
	@BeforeEach
	public void setUp() throws Exception {
		pedestrian = createDeprecatedPedestrian();
	}

	@Override
	@Test
	public void testIncrementNextTargetListIndex() {
		Assertions.assertThrows(IllegalStateException.class, ()->{
			pedestrian.incrementNextTargetListIndex();
		});
	}

	@Override
	@Test
	public void testGetNextTargetId() {
		pedestrian.getTargets().add(3);
		pedestrian.getTargets().add(4);
		assertEquals(3, pedestrian.getNextTargetId());
	}

	@Override
	@Test
	public void testGetNextTargetIdFail() {
		Assertions.assertThrows( NoSuchElementException.class, ()->{
			pedestrian.getTargets().clear();
			pedestrian.getNextTargetId();
		});
	}

	@Override
	@Test
	public void testHasNextTarget() {
		assertFalse(pedestrian.hasNextTarget());
		pedestrian.getTargets().add(0);
		assertTrue(pedestrian.hasNextTarget());
		pedestrian.getTargets().add(1);
		assertTrue(pedestrian.hasNextTarget());
	}

	private Pedestrian createDeprecatedPedestrian() {
		Pedestrian pedestrian = createPedestrian();
		pedestrian.setNextTargetListIndex(-1);
		return pedestrian;
	}
}
