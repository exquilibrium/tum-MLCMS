package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static  org.junit.jupiter.api.Assertions.*;

public class PedestrianEndTimeProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup(){
		processorTestEnv = new PedestrianEndTimeProcessorTestEnv();
		super.setup();
	}

	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

}