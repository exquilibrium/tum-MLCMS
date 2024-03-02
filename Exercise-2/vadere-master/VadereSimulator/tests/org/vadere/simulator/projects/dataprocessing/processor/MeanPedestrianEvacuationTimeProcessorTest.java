package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class MeanPedestrianEvacuationTimeProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new MeanPedestrianEvacuationTimeProcessorTestEnv();
		super.setup();
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