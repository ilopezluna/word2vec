package org.example;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

@Slf4j
public class Load {

	@SneakyThrows
	public static void main(String[] args) {
		log.info("Loading Google News Model...");
		File gModel = new ClassPathResource("GoogleNews-vectors-negative300.bin.gz").getFile();
		Word2Vec vec = WordVectorSerializer.readWord2VecModel(gModel);
		log.info("Closest Words:");
		Collection<String> kingList = vec.wordsNearest(Arrays.asList("king", "woman"),
				Collections.singletonList("queen"), 10);
		log.info("Closest matches to 'king' on Google News: {}", kingList);
	}

}
