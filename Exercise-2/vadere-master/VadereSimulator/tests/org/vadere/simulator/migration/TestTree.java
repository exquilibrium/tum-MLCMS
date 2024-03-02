package org.vadere.simulator.migration;


import com.fasterxml.jackson.databind.JsonNode;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static  org.junit.jupiter.api.Assertions.assertEquals;
import static  org.junit.jupiter.api.Assertions.assertFalse;
import static  org.junit.jupiter.api.Assertions.assertTrue;

import org.vadere.simulator.projects.migration.MigrationException;
import org.vadere.simulator.projects.migration.incident.Tree;
import org.vadere.simulator.projects.migration.incident.incidents.DeleteInArrayIncident;
import org.vadere.simulator.projects.migration.incident.incidents.ExceptionIncident;
import org.vadere.state.util.StateJsonConverter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

public class TestTree {

	private static String json = "";

	@BeforeEach
	public void setUp() throws Exception {
		json = "{\"test\": 1.0, \"persons\":[{\"name\":\"hans\", \"age\": 24},{\"name\":\"bene\", \"age\": 25}], \"house\": {\"street\":\"forest street\", \"number\": 3} }";
	}

	@Test
	public void testPathToString() throws IOException {
		JsonNode jsonNode = StateJsonConverter.deserializeToNode("{\"test\": 1.0}");
		Tree tree = new Tree(jsonNode);
		assertEquals("[a > b > c > d]", tree.pathToString(Arrays.asList("a", "b", "c", "d")));
		assertEquals("[]", tree.pathToString(new ArrayList<>()));
	}

	@Test
	public void testGetNodeByPath() throws IOException {
		JsonNode jsonNode = StateJsonConverter.deserializeToNode(json);
		Tree tree = new Tree(jsonNode);
		assertEquals("street", tree.getNodeByPath(Arrays.asList("house", "street")).getKey());
		tree.getNodeByPath(Arrays.asList("persons")).delete();
	}

	@Test
	public void testRecursiveScan() throws IOException, MigrationException {
		JsonNode jsonNode = StateJsonConverter.deserializeToNode(json);
		Tree tree = new Tree(jsonNode);
		assertEquals("number", tree.recursiveScan(tree.getRoot(), "house", "number", new ExceptionIncident(jsonNode)).get(0).getKey());
		tree.deleteUnrecognizedField("house", "number", new StringBuilder(), new ExceptionIncident(jsonNode));
		assertEquals(new LinkedList<>(), tree.recursiveScan(tree.getRoot(), "house", "number", new ExceptionIncident(jsonNode)));
	}

	@Test
	public void testDeleteNodeInArray() throws IOException, MigrationException {
		JsonNode jsonNode = StateJsonConverter.deserializeToNode(json);
		Tree tree = new Tree(jsonNode);

		assertTrue(tree.getNodeByPath(Arrays.asList("persons")).getJsonNode().get(0).has("name"));
		tree.deleteNodeInArray(Arrays.asList("persons"), "name");
		assertFalse(tree.getNodeByPath(Arrays.asList("persons")).getJsonNode().get(0).has("name"));
		tree.deleteNodeInArray(Arrays.asList("persons"), "name");
		assertFalse(tree.getNodeByPath(Arrays.asList("persons")).getJsonNode().get(0).has("name"));
	}

	@Test
	public void testDeleteNodeInArrayIncident() throws IOException, MigrationException {
		JsonNode jsonNode = StateJsonConverter.deserializeToNode(json);
		DeleteInArrayIncident incident = new DeleteInArrayIncident(Arrays.asList("persons"), "name");
		Tree tree = new Tree(jsonNode);

		assertTrue(incident.applies(tree));
		incident.resolve(tree, new StringBuilder());
		assertFalse(incident.applies(tree));
	}
}
