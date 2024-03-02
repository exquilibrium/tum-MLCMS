package org.vadere.simulator.control;

import org.junit.jupiter.api.Test;
import org.vadere.simulator.control.scenarioelements.SourceController;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.attributes.scenario.SourceTestAttributesBuilder;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.util.geometry.shapes.VPoint;
import org.vadere.util.geometry.shapes.VRectangle;

import java.io.IOException;

import static  org.junit.jupiter.api.Assertions.assertEquals;

public class TestSourceControllerUsingConstantSpawnRate extends TestSourceController {




	/**
	 * Test method for {@link SourceController#update(double)}.
	 */
	@Test
	public void testUpdateEqualStartAndEndTime() throws IOException {

		SourceTestAttributesBuilder builder = new SourceTestAttributesBuilder()
				.setOneTimeSpawn(0);
		initialize(builder);

		first().sourceController.update(0);
		first().sourceController.update(1);
		first().sourceController.update(2);

		assertEquals(1, countPedestrians(0), "wrong pedestrian number");
	}



	/**
	 * Test method for {@link SourceController#update(double)}.
	 */
	@Test
	public void testUpdateEndTimeLarge() throws IOException {

		double startTime = 0.0;
		double endTime = 10.0;
		SourceTestAttributesBuilder builder = new SourceTestAttributesBuilder()
				.setStartTime(startTime).setEndTime(endTime)
				.setDistributionParams(10);
		initialize(builder);

		first().sourceController.update(startTime);
		// one at the beginning
		assertEquals(1, countPedestrians(0), "wrong pedestrian number.");

		first().sourceController.update(endTime);
		// and one at the end
		assertEquals(2, countPedestrians(0), "wrong pedestrian number.");
	}

	/**
	 * Test method for {@link SourceController#update(double)}.
	 */
	@Test
	public void testUpdateSpawnDelayThreeTimes() throws IOException {

		double endTime = 10.0;
		SourceTestAttributesBuilder builder = new SourceTestAttributesBuilder()
				.setStartTime(0).setEndTime(endTime)
				.setDistributionParams(5);
		initialize(builder);

		for (double simTimeInSec = 0; simTimeInSec < endTime * 2; simTimeInSec += 1.0) {
			first().sourceController.update(simTimeInSec);
		}

		assertEquals(3, countPedestrians(0), "wrong pedestrian number.");
	}

	/**
	 * Test method for {@link SourceController#update(double)}.
	 */
	@Test
	public void testUpdateSmallSpawnDelay() throws IOException {

		double endTime = 1.0;

		SourceTestAttributesBuilder builder = new SourceTestAttributesBuilder()
				.setStartTime(0).setEndTime(endTime)
				.setDistributionParams(0.1);
		initialize(builder);

		for (double simTimeInSec = 0; simTimeInSec < endTime * 2; simTimeInSec += 1.0) {
			first().sourceController.update(simTimeInSec);
		}

		assertEquals(11, countPedestrians(0), "Wrong pedestrian number.");
	}

	/**
	 * Test method for {@link SourceController#update(double)}.
	 */
	@Test
	public void testUpdateUseFreeSpaceOnly() throws IOException {

		AttributesAgent attributesAgent = new AttributesAgent();
		SourceTestAttributesBuilder builder = new SourceTestAttributesBuilder()
				.setOneTimeSpawn(0)
				.setSpawnNumber(100)
				.setUseFreeSpaceOnly(true)
				.setSourceDim(new VRectangle(0, 0, attributesAgent.getRadius()*2 + 0.05, attributesAgent.getRadius()*2 + 0.05)); // small source
		initialize(builder);

		for (double simTimeInSec = 0; simTimeInSec < 1000; simTimeInSec += 1.0) {
			first().sourceController.update(simTimeInSec);
		}

		// if the first ped does not move away, there should no more pedestrians
		// be created
		assertEquals(1, countPedestrians(0), "wrong pedestrian number.");

		// now, move the peds away after creating them
		for (double simTimeInSec = 1000; simTimeInSec < 2000; simTimeInSec += 1.0) {
			first().sourceController.update(simTimeInSec);

			VPoint positionFarAway = new VPoint(1000, 1000);
			for (Pedestrian pedestrian : first().topography.getElements(Pedestrian.class)) {
				pedestrian.setPosition(positionFarAway);
			}
		}

		// now, all pedestrians should have been created
		assertEquals(100, countPedestrians(0), "wrong pedestrian number.");
	}



	protected void doUpdates(int source, int number, double startTime, double endTimeExclusive) {
		double timeStep = (endTimeExclusive - startTime) / number;
		for (double t = startTime; t < endTimeExclusive + 1; t += timeStep) {
			sourceTestData.get(source).sourceController.update(t);
		}
	}

	protected void doUpdatesBeamingPedsAway(int source, int number) {
		double start = 10;
		for (double t = start; t < start + number; t += 1) {
			sourceTestData.get(source).sourceController.update(t);
			beamPedsAway(source);
		}
	}

	protected void beamPedsAway(int source) {
		final VPoint positionFarAway = new VPoint(1000, 1000);
		for (Pedestrian pedestrian : sourceTestData.get(source).topography.getElements(Pedestrian.class)) {
			pedestrian.setPosition(positionFarAway);
		}
	}

}
