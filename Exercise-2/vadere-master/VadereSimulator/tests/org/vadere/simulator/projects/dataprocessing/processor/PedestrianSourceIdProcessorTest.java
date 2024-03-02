package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PedestrianSourceIdProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new PedestrianSourceIdProcessorTestEnv();
		super.setup();
	}


	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

}