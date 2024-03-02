package org.vadere.simulator.projects.migration.jsontranformation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.vadere.simulator.utils.reflection.TestJsonNodeExplorer;
import org.vadere.simulator.utils.reflection.TestJsonNodeHelper;
import org.vadere.simulator.utils.reflection.TestResourceHandlerScenario;

public abstract class JsonTransformationTest implements TestJsonNodeExplorer, TestJsonNodeHelper, TestResourceHandlerScenario {

	protected org.vadere.simulator.projects.migration.jsontranformation.JsonTransformationFactory factory = org.vadere.simulator.projects.migration.jsontranformation.JsonTransformationFactory.instance();


	@BeforeEach
	public void init() {
		backupTestDir();
	}

	@AfterEach
	public void cleaUp() {
		loadFromBackup();
	}


}