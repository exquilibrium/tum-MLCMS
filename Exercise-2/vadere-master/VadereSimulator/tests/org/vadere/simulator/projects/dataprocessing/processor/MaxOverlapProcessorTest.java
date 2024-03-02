package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;


public class MaxOverlapProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new MaxOverlapProcessorTestEnv();
		super.setup();
	}


	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}
}