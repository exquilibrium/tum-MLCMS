package org.vadere.simulator.models.groups;

import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.ScenarioElement;

import java.util.LinkedList;
import java.util.List;

public interface Group {
	int getID();

	int getSize();

	boolean isMember(Pedestrian ped);

	List<Pedestrian> getMembers();

	void addMember(Pedestrian ped);

	/**
	 *
	 * @param ped
	 * @return		Retrun True if ped was the last one.
	 */
	boolean removeMember(Pedestrian ped);

	boolean isFull();

	int getOpenPersons();

	boolean equals(Group other);

	void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget);

	IPotentialFieldTarget getPotentialFieldTarget();

	// TODO: --Added functions for SIRGRroup.java
	public void agentTargetsChanged(LinkedList<Integer> targetIds, int agentId);

	public void agentNextTargetSet(double nextSpeed, int agentId);

	public void agentElementEncountered(ScenarioElement element, int agentId);
}
