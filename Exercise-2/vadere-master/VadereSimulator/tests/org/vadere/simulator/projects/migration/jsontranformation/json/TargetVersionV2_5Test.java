package org.vadere.simulator.projects.migration.jsontranformation.json;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;
import org.vadere.simulator.projects.migration.MigrationException;
import org.vadere.simulator.projects.migration.jsontranformation.JsonTransformationTest;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class TargetVersionV2_5Test extends JsonTransformationTest {
    @Override
    public Path getTestDir() {
        return getPathFromResources("/migration/v2_4_to_v2_5");
    }

    @Test
    public void expectNoMigrationOnOVM() throws MigrationException {
        String scenarioFileAsString = getTestFileAsString("v2.4_to_v2.5_Test1.scenario");
        JsonNode oldScenarioAsJson = getJsonFromString(scenarioFileAsString);
        TargetVersionV2_5 transform = factory.getTargetVersionV2_5();
        assertThrows(IllegalArgumentException.class, () -> transform.applyAll(oldScenarioAsJson));

        String oldJsonPath = "scenario/topography/attributesCar";
        pathMustExist(oldScenarioAsJson, oldJsonPath);
    }

    @Test
    public void expectMigrationNo() throws MigrationException {
        String scenarioFileAsString = getTestFileAsString("v2.4_to_v2.5_Test2.scenario");
        JsonNode oldScenarioAsJson = getJsonFromString(scenarioFileAsString);
        TargetVersionV2_5 transform = factory.getTargetVersionV2_5();
        JsonNode newScenarioAsJson = transform.applyAll(oldScenarioAsJson);

        String oldJsonPath = "scenario/topography/attributesCar";
        pathMustNotExist(newScenarioAsJson , oldJsonPath);
    }

}
