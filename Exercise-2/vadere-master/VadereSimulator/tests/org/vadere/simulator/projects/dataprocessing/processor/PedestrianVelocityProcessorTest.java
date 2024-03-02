package org.vadere.simulator.projects.dataprocessing.processor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.state.attributes.processor.AttributesPedestrianVelocityProcessor;

import java.util.LinkedList;

import static  org.junit.jupiter.api.Assertions.*;

public class PedestrianVelocityProcessorTest extends ProcessorTest {

	@BeforeEach
	public void setup(){
		processorTestEnv = new PedestrianVelocityProcessorTestEnv();
		super.setup();
	}

	@Override
	public void assertInit(DataProcessor p) throws NoSuchFieldException, IllegalAccessException {
		AttributesPedestrianVelocityProcessor attr = (AttributesPedestrianVelocityProcessor) p.getAttributes();
		assertEquals(0, p.getData().size(), "Must be zero after init.");
		assertEquals(0, (int) r.valOfField("lastStep"), "Must be zero after init.");
		assertEquals(attr.getBackSteps(), (int)r.valOfField("backSteps"));
		assertEquals(1, ((LinkedList<Double>)r.valOfField("lastSimTimes")).size());
		assertEquals(0.0, ((LinkedList<Double>)r.valOfField("lastSimTimes")).getFirst(), 0.000001);
	}

	@Test
	public void doUpdate() throws Exception {
		// default backstep = 1
		super.doUpdate();
	}

	@Test public void withBackstepTwo() throws Exception {
		AttributesPedestrianVelocityProcessor attr =
				(AttributesPedestrianVelocityProcessor) p.getAttributes();
		attr.setBackSteps(2);
		((PedestrianVelocityProcessorTestEnv) processorTestEnv).loadSimulationStateMocksWithBackstep2();
		p.init(processorTestEnv.getManager());
		super.doUpdate();
	}

	@Test
	public void init() throws Exception {
		super.init();
	}

	@Test
	public void getAttributes() throws Exception {
	}

}