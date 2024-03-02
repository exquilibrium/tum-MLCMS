package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link PedestrianStartTimeProcessorTest} for Test data see {@link
 * PedestrianStartTimeProcessorTestEnv}
 *
 * @author Stefan Schuhbäck
 */
public class PedestrianStartTimeProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new PedestrianStartTimeProcessorTestEnv();
		super.setup();
	}

	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

}