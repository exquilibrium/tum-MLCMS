package org.vadere.util.data.cellgrid;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

import tech.tablesaw.api.Table;
import tech.tablesaw.io.RuntimeIOException;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CellGridTest {

	private File loadTestResource(String path){
		URL resource1 = CellGridTest.class.getResource(path);
		if (resource1 == null){
			fail("Resource not found: " + path);
		}
		return new File(resource1.getFile());
	}

	@Test
	public void loadCache(){
		File path = loadTestResource("/org/vadere/util/data/cellgrid/test001.ffcache");
		Table t = null;
		try {
			t = Table.read().csv(path);
		} catch (RuntimeIOException e) {
			fail("Test file not found");
			return;

		}
		CellGrid cellGrid = new CellGrid(3.0, 3.0, 1.0, new CellState(), 0.0, 0.0);
		assertEquals(t.rowCount(), cellGrid.numPointsX*cellGrid.numPointsY);
//		assertThat(t.rowCount(), equalTo(cellGrid.numPointsX*cellGrid.numPointsY));

		cellGrid.loadFromTable(t);
		compare(t, cellGrid);
	}

	@Test
	public void saveGridToCache(){
		Random rnd = new Random(0);
		CellGrid cellGrid = new CellGrid(3.0, 3.0, 1.0, new CellState(), 0.0, 0.0);
		int maxPathFindingTag = PathFindingTag.values().length;

		// set random data to CellGrid
		for (int row = 0; row < cellGrid.numPointsY; row++) {
			for (int col = 0; col < cellGrid.numPointsX; col++) {
				cellGrid.values[col][row] =
						new CellState(rnd.nextDouble(),
								PathFindingTag.values()[rnd.nextInt(maxPathFindingTag)]);
			}
		}

		compare(cellGrid.asTable(), cellGrid);
	}


	private void compare(Table t, CellGrid cellGrid){
		for (int row = 0; row < cellGrid.numPointsY; row++) {
			for (int col = 0; col < cellGrid.numPointsX; col++) {
				CellState state = cellGrid.values[col][row];
				Table f = t.where(
						t.intColumn("x").isEqualTo(col)
								.and(t.intColumn("y").isEqualTo(row))
				);

//				assertThat(f.rowCount(), equalTo(1));
//				assertThat(f.column("value").get(0), equalTo(state.potential));
//				assertThat(f.column("tag").get(0), equalTo(state.tag.name()));
				assertEquals(1, f.rowCount());
				assertEquals(state.potential, f.column("value").get(0));
				assertEquals(state.tag.name(), f.column("tag").get(0));

			}
		}
	}

}