package org.vadere.simulator.projects;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.simulator.projects.io.IOOutput;
import org.vadere.simulator.projects.io.IOVadere;
import org.vadere.simulator.projects.migration.MigrationAssistant;
import org.vadere.simulator.projects.migration.MigrationOptions;
import org.vadere.simulator.utils.reflection.TestResourceHandlerScenario;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;

import static  org.junit.jupiter.api.Assertions.assertEquals;
import static  org.junit.jupiter.api.Assertions.assertFalse;
import static  org.junit.jupiter.api.Assertions.assertNotNull;
import static  org.junit.jupiter.api.Assertions.assertTrue;
import static  org.junit.jupiter.api.Assertions.fail;

public class ProjectOutputTest implements TestResourceHandlerScenario {

	private VadereProject project;
	private ProjectOutput projectOutput;


	@Override
	public Path getTestDir() {
		Path path = null;
		try {
			path = Paths.get(getClass().getResource("/data/simpleProject").toURI());
		} catch (URISyntaxException e) {
			fail("Test resources not found");
		}
		return path;
	}

	@BeforeEach
	public void setup() throws URISyntaxException, IOException {
		backupTestDir();
		Path projectPath = getTestDir();
		project = IOVadere.readProject(projectPath.toString());
		projectOutput = new ProjectOutput(project);
	}

	@AfterEach
	public void after(){
		loadFromBackup();
	}

	@Test
	public void getAllOutputDirs() throws Exception {
		List<File> out = projectOutput.getAllOutputDirs();
		assertEquals(14, out.size(), "There should be 14 output directories");
	}

	@Test
	public void listSelectedOutputDirs(){
		Optional<SimulationOutput> simOut = projectOutput.getSimulationOutput("testOutput2");
		assertTrue(simOut.isPresent());
		Scenario scenario = simOut.get().getSimulatedScenario();

		List<File> selectedOutputDirs = projectOutput.listSelectedOutputDirs(scenario);
		assertEquals(1, selectedOutputDirs.size(), "There should be one match");
		assertEquals(
				"testOutput2", selectedOutputDirs.get(0).getName(),"The directory name is false");
	}

	@Test
	public void markDirty() {
		projectOutput.markDirty("testOutput2");
		Optional<SimulationOutput> out = projectOutput.getSimulationOutput("testOutput2");
		assertTrue(out.isPresent());
		assertTrue(out.get().isDirty());
	}

	@Test
	public void updateWithDirty() throws IOException, URISyntaxException {
		projectOutput.markDirty("testOutput2");
		Files.delete(project.getOutputDir().resolve(Paths.get("testOutput2", "test_postvis.scenario")));
		projectOutput.update();
		assertFalse(projectOutput.getSimulationOutput("testOutput2").isPresent());
		assertFalse(Files.exists(project.getOutputDir().resolve("testOutput2")));
//
//		//cleanup
//		Path backup = Paths.get(getClass().getResource("/data/testOutput2").toURI());
//		FileUtils.copyDirectoryToDirectory(backup.toFile(), project.getOutputDir().toFile());
//		IOOutput.deleteOutputDirectory(project.getOutputDir().resolve("corrupt/testOutput2").toFile());
	}

	@Test
	public void bi(){
		MigrationAssistant m = MigrationAssistant.getNewInstance(MigrationOptions.defaultOptions());
//		System.out.println(m.migrateScenarioFile());
	}

	@Test
	public void updateWithNew() throws IOException, URISyntaxException {
		Path backup = Paths.get(getClass().getResource("/data/simpleProject/output/testOutput2").toURI());
		FileUtils.copyDirectory(backup.toFile(), project.getOutputDir().resolve("testOutput3").toFile());

		projectOutput.update();
		List<File> out = projectOutput.getAllOutputDirs();
		assertEquals(15, out.size(), "There should be 15 output directories");
		assertTrue(projectOutput.getSimulationOutput("testOutput3").isPresent());

		//cleanup
		IOOutput.deleteOutputDirectory(project.getOutputDir().resolve("testOutput3").toFile());
	}

	@Test
	public void projectOutputSet(){
		VadereProject proj = new VadereProject("test", new LinkedList<>(), Paths.get("."));
		assertNotNull(proj.getProjectOutput());
	}

}
