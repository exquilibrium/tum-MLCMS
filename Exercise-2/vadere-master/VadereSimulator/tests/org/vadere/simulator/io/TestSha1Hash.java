package org.vadere.simulator.io;

import static  org.junit.jupiter.api.Assertions.assertEquals;

import org.apache.commons.codec.digest.DigestUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestSha1Hash {

	@BeforeEach
	public void setUp() throws Exception {}

	@Test
	public void test() {
		String sha1 = DigestUtils.sha1Hex("aff");

		assertEquals("0c05aa56405c447e6678b7f3127febde5c3a9238", sha1);
	}

}
