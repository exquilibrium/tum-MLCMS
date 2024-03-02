package org.vadere.util.io;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.vadere.util.io.parser.JsonLogicParser;
import java.io.IOException;
import java.text.ParseException;

import static  org.junit.jupiter.api.Assertions.assertFalse;
import static  org.junit.jupiter.api.Assertions.assertTrue;

public class TestLogicParser {

	private static String jsonString =
			"{\n" +
					"          \"id\": 6,\n" +
					"          \"shape\": {\n" +
					"            \"x\": 0.5,\n" +
					"            \"y\": 0.5,\n" +
					"            \"width\": 4.1,\n" +
					"            \"height\": 29.0,\n" +
					"            \"type\": \"RECTANGLE\"\n" +
					"          },\n" +
					"          \"spawnDelay\": 2.0,\n" +
					"          \"spawnNumber\": 10,\n" +
					"          \"startTime\": 0.0,\n" +
					"          \"endTime\": 60.0,\n" +
					"          \"spawnAtRandomPositions\": true,\n" +
					"          \"useFreeSpaceOnly\": true,\n" +
					"          \"targetIds\": [\n" +
					"            2,3,4\n" +
					"          ]\n" +
					"        }";

	private JsonNode jsonObject;

	/**
	 * @throws java.lang.Exception
	 */
	@BeforeEach
	public void setUp() throws IOException {
		jsonObject = new ObjectMapper().readTree(jsonString);
	}

	private void assertTrue_(String msg, Boolean result){
		assertTrue(result, msg);
	}

	private void assertFalse_(String msg, Boolean result){
		assertFalse(result, msg);
	}

	@Test
	public void testSubSet() throws ParseException {
		// spacing test
		assertTrue_("id:{5,6,7}", new JsonLogicParser("id:{5,6,7}").parse().test(jsonObject));
		assertTrue_("id:{5,6,7}", new JsonLogicParser("id:{5,6,7}").parse().test(jsonObject));
		assertTrue_("id: {5,6,7}", new JsonLogicParser("id: {5,6,7}").parse().test(jsonObject));
		assertTrue_("id : {5,6,7}", new JsonLogicParser("id : {5,6,7}").parse().test(jsonObject));
		assertTrue_("id :{5,6,7}", new JsonLogicParser("id :{5,6,7}").parse().test(jsonObject));
		// spacing test end

		assertTrue_("shape.x:{0.5,0.7,0.1}", new JsonLogicParser("shape.x:{0.5,0.7,0.1}").parse().test(jsonObject));
		assertTrue_("targetIds:{2,3,4,5.3}", new JsonLogicParser("targetIds:{2,3,4,5.3}").parse().test(jsonObject));
		assertTrue_("targetIds:{1,2,3,4,5}", new JsonLogicParser("targetIds:{1,2,3,4,5}").parse().test(jsonObject));
		assertTrue_("targetIds:{1,2.0,3,4,5}", new JsonLogicParser("targetIds:{1,2.0,3,4,5}").parse().test(jsonObject));

		assertFalse_("id:{1,2,3}", new JsonLogicParser("id:{1,2,3}").parse().test(jsonObject));
		assertFalse_("shape.x:{0.55,0.7,0.1}", new JsonLogicParser("shape.x:{0.55,0.7,0.1}").parse().test(jsonObject));
		assertFalse_("targetIds:{2,3,5,6}", new JsonLogicParser("targetIds:{2,3,5,6}").parse().test(jsonObject));
		assertFalse_("targetIds:{9}", new JsonLogicParser("targetIds:{9}").parse().test(jsonObject));
	}

	@Test
	public void testSuperSet() throws ParseException {
		// spacing test
		assertTrue_("{6}: id", new JsonLogicParser("{6}: id").parse().test(jsonObject));
		assertTrue_("{6} :id", new JsonLogicParser("{6} :id").parse().test(jsonObject));
		assertTrue_("{6} : id", new JsonLogicParser("{6} : id").parse().test(jsonObject));
		assertTrue_("{6}:id", new JsonLogicParser("{6}:id").parse().test(jsonObject));
		// spacing test end

		assertTrue_("{0.5}:shape.x", new JsonLogicParser("{0.5}:shape.x").parse().test(jsonObject));
		assertTrue_("{2,3,4}:targetIds", new JsonLogicParser("{2,3,4}:targetIds").parse().test(jsonObject));
		assertTrue_("{2,3.0,4}:targetIds", new JsonLogicParser("{2,3.0,4}:targetIds").parse().test(jsonObject));
		assertTrue_("{3.0,4}:targetIds", new JsonLogicParser("{3.0,4}:targetIds").parse().test(jsonObject));

		assertFalse_("{6,7}:id", new JsonLogicParser("{6,7}:id").parse().test(jsonObject));
		assertFalse_("{7}:id", new JsonLogicParser("{7}:id").parse().test(jsonObject));
		assertFalse_("{3.0,4,2,5.0}:targetIds", new JsonLogicParser("{3.0,4,2,5.0}:targetIds").parse().test(jsonObject));
	}

	@Test
	public void testEquals() throws ParseException {
		// spacing test
		assertTrue_("id ==6", new JsonLogicParser("id ==6").parse().test(jsonObject));
		assertTrue_("id== 6", new JsonLogicParser("id== 6").parse().test(jsonObject));
		assertTrue_("id == 6", new JsonLogicParser("id == 6").parse().test(jsonObject));
		assertTrue_("id= =6", new JsonLogicParser("id= =6").parse().test(jsonObject));
		assertTrue_("id==6", new JsonLogicParser("id==6").parse().test(jsonObject));
		// spacing end

		assertTrue_("shape.x==0.5", new JsonLogicParser("shape.x==0.5").parse().test(jsonObject));
		assertTrue_("shape.type==RECTANGLE", new JsonLogicParser("shape.type==RECTANGLE").parse().test(jsonObject));

		assertFalse_("id==8", new JsonLogicParser("id==8").parse().test(jsonObject));
		assertFalse_("shape.x==0.7", new JsonLogicParser("id==8").parse().test(jsonObject));
		assertFalse_("shape.type==POLYGON", new JsonLogicParser("shape.type==POLYGON").parse().test(jsonObject));
	}

	@Test
	public void testNotEquals() throws ParseException {
		// spacing test
		assertFalse_("id!= 6", new JsonLogicParser("id!= 6").parse().test(jsonObject));
		assertFalse_("id !=6", new JsonLogicParser("id !=6").parse().test(jsonObject));
		assertFalse_("id! =6", new JsonLogicParser("id! =6").parse().test(jsonObject));
		assertFalse_("id != 6", new JsonLogicParser("id != 6").parse().test(jsonObject));
		assertFalse_("id ! = 6", new JsonLogicParser("id ! = 6").parse().test(jsonObject));
		assertFalse_("id!=6", new JsonLogicParser("id!=6").parse().test(jsonObject));
		// spacing end

		assertFalse_("shape.x!=0.5", new JsonLogicParser("shape.x!=0.5").parse().test(jsonObject));
		assertFalse_("shape.type!=RECTANGLE", new JsonLogicParser("shape.type!=RECTANGLE").parse().test(jsonObject));

		assertTrue_("id!=8", new JsonLogicParser("id!=8").parse().test(jsonObject));
		assertTrue_("shape.x!=0.7", new JsonLogicParser("id!=8").parse().test(jsonObject));
		assertTrue_("shape.type!=POLYGON", new JsonLogicParser("shape.type!=POLYGON").parse().test(jsonObject));
	}

	@Test
	public void testNot() throws ParseException {
		// spacing test
		assertFalse_("!(id==6)", new JsonLogicParser("!(id==6)").parse().test(jsonObject));
		assertFalse_("! (id==6)", new JsonLogicParser("! (id==6)").parse().test(jsonObject));
		assertFalse_("! (id= =6)", new JsonLogicParser("! (id= =6)").parse().test(jsonObject));
		// spacing end

		assertFalse_("!(shape.x==0.5)", new JsonLogicParser("!(shape.x==0.5)").parse().test(jsonObject));
		assertFalse_("!(shape.type==RECTANGLE)",
				new JsonLogicParser("!(shape.type==RECTANGLE)").parse().test(jsonObject));

		assertTrue_("!(id==8)", new JsonLogicParser("!(id==8)").parse().test(jsonObject));
		assertTrue_("!(shape.x==0.7)", new JsonLogicParser("!(shape.x==0.7)").parse().test(jsonObject));
		assertTrue_("!(shape.type==POLYGON)", new JsonLogicParser("!(shape.type==POLYGON)").parse().test(jsonObject));
	}

	@Test
	public void testGreaterThan() throws ParseException {
		// spacing test
		assertTrue_("id> 5", new JsonLogicParser("id> 5").parse().test(jsonObject));
		assertTrue_("id >5", new JsonLogicParser("id >5").parse().test(jsonObject));
		assertTrue_("id > 5", new JsonLogicParser("id > 5").parse().test(jsonObject));
		assertTrue_("id>5", new JsonLogicParser("id>5").parse().test(jsonObject));
		// spacing end

		assertTrue_("shape.x>0.4", new JsonLogicParser("shape.x>0.4").parse().test(jsonObject));

		assertFalse_("id>6", new JsonLogicParser("id>6").parse().test(jsonObject));
		assertFalse_("id>8", new JsonLogicParser("id>8").parse().test(jsonObject));
		assertFalse_("shape.x>0.7", new JsonLogicParser("shape.x>0.7").parse().test(jsonObject));
	}

	@Test
	public void testGreaterThanOrEquals() throws ParseException {
		assertTrue_("id>=5", new JsonLogicParser("id>=5").parse().test(jsonObject));
		assertTrue_("id>=6", new JsonLogicParser("id>=6").parse().test(jsonObject));
		assertTrue_("shape.x>=0.4", new JsonLogicParser("shape.x>=0.4").parse().test(jsonObject));
		assertTrue_("shape.x>=0.5", new JsonLogicParser("shape.x>=0.5").parse().test(jsonObject));

		assertFalse_("id>=8", new JsonLogicParser("id>=8").parse().test(jsonObject));
		assertFalse_("shape.x>=0.7", new JsonLogicParser("id>=8").parse().test(jsonObject));
	}

	@Test
	public void testSmallerThan() throws ParseException {
		assertTrue_("id<8", new JsonLogicParser("id<8").parse().test(jsonObject));
		assertTrue_("shape.x<0.6", new JsonLogicParser("shape.x<0.6").parse().test(jsonObject));

		assertFalse_("id<5", new JsonLogicParser("id<5").parse().test(jsonObject));
		assertFalse_("id<4", new JsonLogicParser("id<4").parse().test(jsonObject));
		assertFalse_("shape.x<0.4", new JsonLogicParser("shape.x<0.4").parse().test(jsonObject));
	}

	@Test
	public void testSmallerThanOrEquals() throws ParseException {
		assertTrue_("id<=6", new JsonLogicParser("id<=6").parse().test(jsonObject));
		assertTrue_("id<=7", new JsonLogicParser("id<=7").parse().test(jsonObject));
		assertTrue_("shape.x<=0.5", new JsonLogicParser("shape.x<=0.5").parse().test(jsonObject));
		assertTrue_("shape.x<=0.6", new JsonLogicParser("shape.x<=0.6").parse().test(jsonObject));

		assertFalse_("id<=5", new JsonLogicParser("id<=5").parse().test(jsonObject));
		assertFalse_("shape.x<=0.3", new JsonLogicParser("shape.x<=0.3").parse().test(jsonObject));
	}

	@Test
	public void testComplexExpression() throws ParseException {
		assertTrue_("true", new JsonLogicParser("true").parse().test(jsonObject));
		assertTrue_("id<=6 && id>3", new JsonLogicParser("id<=6 && id>3").parse().test(jsonObject));
		assertTrue_("(id<=6 && id>8) || id:{5,6,7}",
				new JsonLogicParser("(id<=6 && id>8) || id:{5,6,7}").parse().test(jsonObject));
		assertTrue_("(id<=6 && id>4 && id<7 && spawnDelay==2.0) && {6}:id",
				new JsonLogicParser("(id<=6 && id>4 && id<7 && spawnDelay==2.0) && {6}:id").parse().test(jsonObject));
		assertTrue_("(id<=6 || id>8 && id<7 || spawnDelay==7.0) && {6}:id",
				new JsonLogicParser("(id<=6 || id>8 && id<7 || spawnDelay==7.0) && {6}:id").parse().test(jsonObject));

		assertFalse_("false", new JsonLogicParser("false").parse().test(jsonObject));
		assertFalse_("id<=6 && id>6", new JsonLogicParser("id<=6 && id>6").parse().test(jsonObject));
		assertFalse_("(id<=5 && id>8) || id:{1,2,3}",
				new JsonLogicParser("(id<=5 && id>8) || id:{1,2,3}").parse().test(jsonObject));
		assertFalse_("(id<=6 && id>4 && id<7 && spawnDelay==2.0) && {6,7}:id",
				new JsonLogicParser("(id<=6 && id>4 && id<7 && spawnDelay==2.0) && {6,7}:id").parse().test(jsonObject));
		assertFalse_("(id<=5 || id>8 && id<7 || spawnDelay==7.0) && {6}:id",
				new JsonLogicParser("(id<=5 || id>8 && id<7 || spawnDelay==7.0) && {6}:id").parse().test(jsonObject));

		assertTrue_("!(id<=5 || id>8 && id<7 || spawnDelay==7.0) && {6}:id",
				new JsonLogicParser("!(id<=5 || id>8 && id<7 || spawnDelay==7.0) && {6}:id").parse().test(jsonObject));
	}

	@Test
	public void testAnd() throws ParseException {
		assertTrue_("shape.x<1.4 && shape.x>0.2",
				new JsonLogicParser("shape.x<1.4 && shape.x>0.2").parse().test(jsonObject));
		assertFalse_("shape.x<1.4 && shape.x>0.6",
				new JsonLogicParser("shape.x < 1.4 && shape.x>0.6").parse().test(jsonObject));
	}
}
