/* ### Task 4-5, 5: added this modified copy of FootStepProcessor,
changes are marked similiar to this comment */
// <<<<<<<<<<!!!>>>>>>>>>>

package org.vadere.simulator.projects.dataprocessing.processor;

import org.vadere.annotation.factories.dataprocessors.DataProcessorClass;
import org.vadere.simulator.control.simulation.SimulationState;
import org.vadere.simulator.models.groups.cgm.CentroidGroupModel;
import org.vadere.simulator.models.groups.sir.SIRGroupModel;
import org.vadere.simulator.projects.dataprocessing.datakey.EventtimePedestrianIdKey;
import org.vadere.simulator.projects.dataprocessing.processor.util.ModelFilter;
import org.vadere.util.logging.Logger;

@DataProcessorClass
public class SecondGroupIDProcessor extends DataProcessor<EventtimePedestrianIdKey, Integer> implements ModelFilter {

	private static Logger logger = Logger.getLogger(PedestrianGroupIDProcessor.class);

	public SecondGroupIDProcessor(){
		super("groupId");
	}

	@Override
	protected void doUpdate(SimulationState state) {
		getModel(state, CentroidGroupModel.class).ifPresent(m -> { // find CentroidGroupModel
			CentroidGroupModel model = (CentroidGroupModel)m;
			model.getGroupsById().forEach((gId, group) -> {	// for each group
				group.getMembers().forEach(ped -> {			// for each member in group
					/* ### Task 4-5, 5: changed this so it always adds the
					pedestrian even when it did not move and registers the current sim time*/
					// <<<<<<<<<<!!!>>>>>>>>>>
					this.putValue(new EventtimePedestrianIdKey(state.getSimTimeInSec(), ped.getId()), gId);
					// <<<<<<<<<<!!!>>>>>>>>>>
				});
			});
		});

		getModel(state, SIRGroupModel.class).ifPresent(m -> { // find SIRGroupModel
			SIRGroupModel model = (SIRGroupModel)m;
			model.getGroupsById().forEach((gId, group) -> {	// for each group
				group.getMembers().forEach(ped -> {			// for each member in group
					/* ### Task 4-5, 5: changed this so it always adds the
					pedestrian even when it did not move and registers the current sim time*/
					// <<<<<<<<<<!!!>>>>>>>>>>
					this.putValue(new EventtimePedestrianIdKey(state.getSimTimeInSec(), ped.getId()), gId);
					// <<<<<<<<<<!!!>>>>>>>>>>
				});
			});
		});
	}

	public String[] toStrings(EventtimePedestrianIdKey key){
		Integer i = this.getValue(key);
		if (i == null) {
			logger.warn(String.format("FootstepGroupIDProcessor does not have Data for Key: %s",
					key.toString()));
			i = -1;
		}

		return new String[]{Integer.toString(i)};
	}
}
// <<<<<<<<<<!!!>>>>>>>>>>