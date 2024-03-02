package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static  org.junit.jupiter.api.Assertions.*;

public class PedestrianMeanFlowProcessorTest extends ProcessorTest {

	@Override
	@BeforeEach
	public void setup(){
		processorTestEnv = new PedestrianFlowProcessorTestEnv();
		super.setup();
	}

	@Override
	@Test
	public void doUpdate() throws Exception {
		super.doUpdate();
	}

	@Override
	@Test
	public void init() throws Exception {
		super.init();
	}

}