package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class AreaDensityVoronoiProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new AreaDensityVoronoiProcessorTestEnv();
		super.setup();
	}

	@Test
	public void doUpdate() throws Exception {
		AreaDensityVoronoiProcessorTestEnv env = (AreaDensityVoronoiProcessorTestEnv) processorTestEnv;
		env.loadOneCircleEvent();
		super.doUpdate();
	}

	/**
	 * This will fail. The implementation fails to create the write segmentation if all
	 * Pedestrian are collinear.
	 */
	@Test
	@Disabled
	public void withCollinear() throws Exception {
		AreaDensityVoronoiProcessorTestEnv env = (AreaDensityVoronoiProcessorTestEnv) processorTestEnv;
		env.loadCollinearSetup();
		super.doUpdate();
	}

	@Test
	@Disabled
	public void init() throws Exception {
		super.init();
	}

}