package org.example;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;

@Slf4j
public class Generate {

	@SneakyThrows
	public static void main(String[] args) {
		String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
		log.info("Load & Vectorize Sentences....");
		SentenceIterator iter = new BasicLineIterator(filePath);
		TokenizerFactory t = new DefaultTokenizerFactory();

		/*
		 * CommonPreprocessor will apply the following regex to each token:
		 * [\d\.:,"'\(\)\[\]|/?!;]+ So, effectively all numbers, punctuation symbols and
		 * some special symbols are stripped off. Additionally it forces lower case for
		 * all tokens.
		 */
		t.setTokenPreProcessor(new CommonPreprocessor());

		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5)
			.iterations(1)
			.layerSize(100)
			.seed(42)
			.windowSize(5)
			.iterate(iter)
			.tokenizerFactory(t)
			.build();

		log.info("Fitting Word2Vec model....");
		vec.fit();

		log.info("Writing word vectors to text file....");

		// Write word vectors to file
		// WordVectorSerializer.writeWord2VecModel(vec, "target/output.txt");

		// Prints out the closest 10 words to "day". An example on what to do with these
		// Word Vectors.
		log.info("Closest Words:");
		Collection<String> lst = vec.wordsNearest("day", 10);
		System.out.println("10 Words closest to 'day': " + lst);
	}

}
