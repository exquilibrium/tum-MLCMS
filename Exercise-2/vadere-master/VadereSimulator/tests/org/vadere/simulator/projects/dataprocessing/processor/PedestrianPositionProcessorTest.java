package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.simulator.utils.reflection.ReflectionHelper;


/**
 * Tests for {@link PedestrianPositionProcessor} for Test data see {@link
 * PedestrianPositionProcessorTestEnv}
 *
 * @author Stefan Schuhbäck
 */
public class PedestrianPositionProcessorTest extends ProcessorTest {


	@BeforeEach
	public void setup() {
		processorTestEnv = new PedestrianPositionProcessorTestEnv();
		processorTestEnv.loadDefaultSimulationStateMocks();
		processorTestEnv.init();
		p = processorTestEnv.getTestedProcessor();
		r = ReflectionHelper.create(p);
	}

	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

	@Test
	public void init() throws Exception {
		super.init();
	}


}