package org.vadere.util.io;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.util.data.Table;
import org.vadere.util.io.TableReader;

import java.util.Formatter;
import java.util.Locale;
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static  org.junit.jupiter.api.Assertions.assertEquals;

public class TestIOUtils {

	private String tableDataString1 = "0,1,2.3; x" + System.lineSeparator() + "1,2,2.42; x";
	private String tableDataFormat1 = "%d,%d,%f; %s";
	private String tableDataVars1 = "a,b,c,d";

	private String hugeTable;
	private String dataFormat2;
	private Random random = new Random();
	private int lineNumber;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeEach
	public void setUp() throws Exception {
		tableDataString1 = "0,1,2.3; x" + System.lineSeparator() + "1,2,2.42; x";
		tableDataFormat1 = "%d,%d,%f; %s";
		tableDataVars1 = "a,b,c,d";

		random = new Random();
		lineNumber = 0;
	}

	@Test
	public void testReadTableFormat1() {
		String dataFormat = "%f %f; id(%d) %s";
		String headerFormat = dataFormat.replaceAll("%.", "%s");

		TableReader reader = new TableReader(dataFormat);

		String[] headline =
				reader.readHeadLine(Stream.generate(() -> generateLine(dataFormat, headerFormat)).limit(401));
		lineNumber = 0;
		Table t = reader.readTable(Stream.generate(() -> generateLine(dataFormat, headerFormat)).limit(401), headline);

		Consumer<Integer> checkRow = i -> {
			assertEquals((i + 1) * 10.0, t.getEntry("x", i), "wrong data in the table, pos x");
			assertEquals((i + 1) * 3.0, t.getEntry("y", i), "wrong data in the table, pos y");
			assertEquals((i + 1), t.getEntry("id", i), "wrong data in the table, pos id");
			assertEquals("s" + (i + 1) + "d", t.getEntry("name", i), "wrong data in the table, pos name");
		};

		Stream.iterate(0, i -> i + 1).limit(400).forEach(checkRow);
	}

	@Test
	public void testReadTable() {
		TableReader reader = new TableReader(tableDataFormat1);
		Table t =
				reader.readTable(Stream.of(tableDataString1.split(System.lineSeparator())), false, "a", "b", "c", "d");

		assertEquals(2, t.size(), "table size does not match.");
		assertEquals(0, t.getEntry("a", 0), "wrong data in the table, pos a");
		assertEquals(1, t.getEntry("b", 0), "wrong data in the table, pos b");
		assertEquals(2.3, t.getEntry("c", 0), "wrong data in the table, pos c");
		assertEquals("x", t.getEntry("d", 0), "wrong data in the table, pos d");
	}


	private String generateLine(final String dataFormat, final String headerFormat) {
		String result = "";
		if (lineNumber == 0) {
			result = new Formatter(Locale.ENGLISH).format(headerFormat, "x", "y", "id", "name").toString();
		} else {
			result = new Formatter(Locale.ENGLISH)
					.format(dataFormat, lineNumber * 10.0, lineNumber * 3.0, lineNumber, "s" + lineNumber + "d")
					.toString();
		}
		lineNumber++;
		return result;
	}
}
