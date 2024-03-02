package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class EvacuationTimeProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new EvacuationTimeProcessorTestEnv();
		super.setup();
	}


	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

	@Test
	public void doUpdateNaN() throws Exception {
		((EvacuationTimeProcessorTestEnv) processorTestEnv).loadSimulationStateMocksNaN();
		super.doUpdate();
	}

	@Test
	public void init() throws Exception {
		super.init();
	}

}