package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.state.attributes.processor.AttributesAreaProcessor;
import org.vadere.util.geometry.shapes.VRectangle;

import static  org.junit.jupiter.api.Assertions.assertEquals;

public class AreaSpeedProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup() {
		processorTestEnv = new AreaSpeedProcessorTestEnv();
		super.setup();
	}

	@Override
	public void assertInit(DataProcessor p) throws NoSuchFieldException, IllegalAccessException {
		assertEquals(0, p.getData().size(), "Must be zero after init.");
		assertEquals(0, (int) r.valOfField("lastStep"), "Must be zero after init.");
		AttributesAreaProcessor attr = (AttributesAreaProcessor) p.getAttributes();
	}

	@Test
	public void doUpdate() throws Exception {
		AttributesAreaProcessor attr = (AttributesAreaProcessor) p.getAttributes();
		processorTestEnv.init();
		super.doUpdate();
	}

	@Test
	public void init() throws Exception {
		super.init();
	}

}