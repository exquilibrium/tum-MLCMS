package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static  org.junit.jupiter.api.Assertions.*;

@Disabled
public class BonnMotionTrajectoryProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup(){
		processorTestEnv = new BonnMotionTrajectoryProcessorTestEnv();
		super.setup();
	}

	@Test
	public void doUpdate() throws Exception{
		//DefaultSimulationStateMocks
		super.doUpdate();
	}

}