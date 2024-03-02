package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class PedestrianTargetIdProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup(){
		processorTestEnv = new PedestrianTargetIdProcessorTestEnv();
		//int and loadFromFilesystem ProcessorTestEnv
		super.setup();
	}

	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

}