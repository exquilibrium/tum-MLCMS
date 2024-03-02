package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PedestrianStateProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new PedestrianStateProcessorTestEnv();
		super.setup();
	}

	@Test
	public void init() throws Exception {
		super.init();
	}


	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

}