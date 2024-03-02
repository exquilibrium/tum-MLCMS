package org.vadere.state.attributes;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestAttributesCloneable {
	
	private static class NotCloneableClass {
		@Override
		public Object clone() throws CloneNotSupportedException {
			return super.clone(); // throws exception because it does not implement Cloneable
		}
	}

	@Test
	public void testNotCloneableClass() throws CloneNotSupportedException {
		Assertions.assertThrows(CloneNotSupportedException.class, ()->{
			new NotCloneableClass().clone();
		});
	}

}
