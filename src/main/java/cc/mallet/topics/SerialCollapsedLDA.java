package cc.mallet.topics;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.apache.commons.lang.NotImplementedException;

import cc.mallet.configuration.LDAConfiguration;
// import cc.mallet.topics.SimpleLDA;
// import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.LoggingUtils;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;
import cc.mallet.util.Timing;

public class SerialCollapsedLDA extends SimpleLDA implements LDAGibbsSampler {

	private static final long serialVersionUID = 7533987649605469394L;
	private static Logger logger = MalletLogger.getLogger(SerialCollapsedLDA.class.getName());
	LDAConfiguration config;
	int currentIteration = 0 ;
	private int startSeed;
	boolean abort = false;

	// The original training data
	InstanceList trainingData;

	// Used for inefficiency calculations
	int [][] topIndices = null;

	public SerialCollapsedLDA(LDAConfiguration config) {
		super(config.getNoTopics(LDAConfiguration.NO_TOPICS_DEFAULT),
				config.getAlpha(LDAConfiguration.ALPHA_DEFAULT)*config.getNoTopics(LDAConfiguration.NO_TOPICS_DEFAULT),
				config.getBeta(LDAConfiguration.BETA_DEFAULT),
				new Randoms(config.getSeed(LDAConfiguration.SEED_DEFAULT))
				);
		setConfiguration(config);
		printLogLikelihood = false;
		showTopicsInterval = config.getTopicInterval(LDAConfiguration.TOPIC_INTER_DEFAULT);
	}
	
	@Override
	public void setRandomSeed(int seed) {
		super.setRandomSeed(seed);
		startSeed = seed;
	}

	public int getStartSeed() {
		return startSeed;
	}

	public SerialCollapsedLDA(int numberOfTopics) {
		super(numberOfTopics);
	}

	public SerialCollapsedLDA(int numberOfTopics, double alpha, double beta) {
		super(numberOfTopics, alpha*numberOfTopics, beta);
	}

	public SerialCollapsedLDA(int numberOfTopics, double alpha, double beta,
			Randoms random) {
		super(numberOfTopics, alpha*numberOfTopics, beta, random);
	}

	@Override
	public void sample (int iterations) throws IOException {
		boolean computeLikelihood = config.computeLikelihood();
		String loggingPath = config.getLoggingUtil().getLogDir().getAbsolutePath();
		double logLik; 
		String tw; 
		if (computeLikelihood) {
			logLik = modelLogLikelihood();
			tw = topWords(wordsPerTopic);
			LDAUtils.logLikelihoodToFile(logLik, 0, tw, loggingPath, logger);
		}
		
		int [] printFirstNDocs = config.getPrintNDocsInterval();
		int nDocs = config.getPrintNDocs();
		int [] printFirstNTopWords = config.getPrintNTopWordsInterval();
		int nWords = config.getPrintNTopWords();
		
		int [] defaultVal = {-1};
		int [] output_interval = config.getIntArrayProperty("diagnostic_interval",defaultVal);
		File binOutput = null;
		if(output_interval.length>1||printFirstNDocs.length>1||printFirstNTopWords.length>1) {
			binOutput = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/binaries");
		}
		
		boolean savePhi = config.getSavePhi();
		int startDiagnostic = config.getStartDiagnostic(LDAConfiguration.START_DIAG_DEFAULT);

		// --- plda: added on July 10, 2021 --- 
		File asciiOutput = null;
		if (output_interval.length > 1 || printFirstNDocs.length > 1 || printFirstNTopWords.length > 1 || startDiagnostic >= 1) {
			asciiOutput = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/ascii");
		}

		long maxExecTimeMillis = TimeUnit.MILLISECONDS.convert(config.getMaxExecTimeSeconds(LDAConfiguration.EXEC_TIME_DEFAULT), TimeUnit.SECONDS); 
		System.out.println("\nMax exec time (millseconds) = " + maxExecTimeMillis + "\n");

		// ----------- plda --------------

		System.out.println("\nDocument sampling hyperparameter alpha = " + alpha);
		System.out.println("Topic sampling hyperparameter beta = " + beta + "\n");

		System.out.println("Number of topics = " + numTopics);
		System.out.println("Number of unique words = " + numTypes);


		long zSamplingTime = 0; 

		for (int iteration = 1; iteration <= iterations && !abort; iteration++) {
			currentIteration = iteration;

			long iterationStart = System.currentTimeMillis();

			// Loop over every document in the corpus
			for (int doc = 0; doc < data.size(); doc++) {
				FeatureSequence tokenSequence =
						(FeatureSequence) data.get(doc).instance.getData();
				LabelSequence topicSequence =
						(LabelSequence) data.get(doc).topicSequence;

				sampleTopicsForOneDoc (tokenSequence, topicSequence);
			}

			long elapsedMillis = System.currentTimeMillis();
			zSamplingTime += (elapsedMillis - iterationStart); // accumulate the time 
			logger.fine(iteration + "\t" + (elapsedMillis - iterationStart) + "ms\t");

			if(config!= null) { 
				config.getLoggingUtil().logTiming(new Timing(iterationStart,elapsedMillis,"CollapsedSample_Z"));
			}

			if(output_interval.length == 2 && iteration >= output_interval[0] && iteration <= output_interval[1]) {
				LDAUtils.writeBinaryIntMatrix(typeTopicCounts, iteration, numTypes, numTopics, binOutput.getAbsolutePath() + "/Serial_N");
				LDAUtils.writeBinaryIntMatrix(LDAUtils.getDocumentTopicCounts(data, numTopics), iteration, data.size(), numTopics, binOutput.getAbsolutePath() + "/Serial_M");
			}

			if (showTopicsInterval > 0 && iteration % showTopicsInterval == 0 && computeLikelihood) {
				if(config!= null) { 
					logLik = modelLogLikelihood();
					tw = topWords (wordsPerTopic);
					LDAUtils.logLikelihoodToFile(logLik,iteration,tw,loggingPath,logger);
				}
			}

			if( printFirstNTopWords.length > 1 && LDAUtils.inRangeInterval(iteration, printFirstNTopWords)) {
				// Assign these once
				if(topIndices==null) {
					topIndices = LDAUtils.getTopWordIndices(nWords, numTypes, numTopics, typeTopicCounts, alphabet);
				}
				double [][] phi = LDAUtils.drawDirichlets(typeTopicCounts);
				LDAUtils.writeBinaryDoubleMatrixIndices(LDAUtils.transpose(phi), iteration, binOutput.getAbsolutePath() + "/Selected_Phi_KxV", topIndices);
			}

			// Start changes on Jan 14, 2022 ---------
			int numDocuments = data.size();
			if (startDiagnostic > 0 && iteration >= startDiagnostic) {

				// Augmented Gibbs sampling of \theta
				int[][] docTopicCounts = LDAUtils.getDocumentTopicCounts(data, numTopics, numDocuments);
				double[][] theta = LDAUtils.drawDirichlets(docTopicCounts, alpha);

				// Compute Quantiy C_s
				StringBuilder strMinDocsDist = new StringBuilder();
				strMinDocsDist.append(iteration);
				for (int d = 0; d < numDocuments; d++) {
					double minDocsDist = 1e+20;
					for (int dd = 0; dd < numDocuments; dd++) {
						if (d != dd) {
							// compute Euclidean distance between \theta_d, \theta_s
							double docDist = 0.0;
							for (int k = 0; k < numTopics; k++) {
								docDist += Math.pow(theta[d][k] - theta[dd][k], 2.0);
							}
							docDist = Math.sqrt(docDist);
							if (docDist < minDocsDist) {
								minDocsDist = docDist;
							}
						}
					}
					strMinDocsDist.append(",");
					strMinDocsDist.append(minDocsDist);
				}

				// TODO: Find an efficient way to save the distances
				String docDistFile = loggingPath + "/min_doc_distances.csv";
				try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(docDistFile, true)))) {
					out.println(strMinDocsDist.toString());
				} catch (IOException e) {
					e.printStackTrace();
					System.err.println("Could not write minimum topic distance file");
				}

				if (printFirstNDocs.length > 1 && LDAUtils.inRangeInterval(iteration, printFirstNDocs)) {
					// --- plda: added on July 10, 2021 ---
					String fn = String.format(
							asciiOutput.getAbsolutePath() + "/Theta_DxK"
									+ "_" + nDocs
									+ "_" + numTopics
									+ "_%05d.csv",
							iteration);

					if (numDocuments > nDocs) {
						double[][] thetaNew = new double[nDocs][numTopics];
						for (int d = 0; d < nDocs; d++) {
							thetaNew[d] = theta[d];
						}
						LDAUtils.writeASCIIDoubleMatrix(thetaNew, fn, ",");
					} else {
						LDAUtils.writeASCIIDoubleMatrix(theta, fn, ",");
					}
					// ----------- plda --------------
				}

				// Augmented Gibbs sampling of \phi
				double[][] phi = LDAUtils.drawDirichlets(typeTopicCounts, beta);

				// Compute Quantity B 
				double minTopicDist = 1e+20;
				int K = phi.length; // number of topics
				int V = phi[0].length; // number of unique words in the vocabulary
				for (int i = 0; i < K; i++) {
					for (int j = 0; j < i; j++) {
						// compute Euclidean distance between \phi_i, \phi_j
						double topicDist = 0.0;
						for (int v = 0; v < V; v++) {
							topicDist += Math.pow(phi[i][v] - phi[j][v], 2.0);
						}
						topicDist = Math.sqrt(topicDist);
						if (topicDist < minTopicDist) {
							minTopicDist = topicDist;
						}
					}
				}

				// TODO: Find an efficient way to save the distances
				String topicDistFile = loggingPath + "/min_topic_distances.csv";
				try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(topicDistFile, true)))) {
					out.println(iteration + "," + minTopicDist);
				} catch (IOException e) {
					e.printStackTrace();
					System.err.println("Could not write minimum topic distance file");
				}

				// Saves the phi matrix in every diagnostic iteration
				if (savePhi) {
					String fn = String.format(
							asciiOutput.getAbsolutePath() + "/Phi_KxV"
									+ "_" + K
									+ "_" + V
									+ "_%05d.csv",
							iteration);
					LDAUtils.writeASCIIDoubleMatrix(phi, fn, ",");
				}
			}
			// End of changes on Jan 14, 2022 ---------


			if (zSamplingTime >= maxExecTimeMillis){
				break; 
			}
		}

		System.out.println(
			"\nCGS: z sampling time (milliseconds) = " + 
			zSamplingTime + 
			" (iteration: " + 
			currentIteration + 
			") \n"
			);

	}

	/**
	 * @return the config
	 */
	public LDAConfiguration getConfig() {
		return config;
	}

	@Override
	/**
	 * @param config the config to set
	 */
	public void setConfiguration(LDAConfiguration config) {
		this.config = config;
	}

	/* 
	 * Uses AD-LDA logLikelihood calculation
	 *  
	 * Here we override SimpleLDA's original likelihood calculation and use the
	 * AD-LDA logLikelihood calculation. It's unclear to me which is "correct"
	 * With this approach all models likelihoods are calculated the same way
	 */
	@Override
	public double modelLogLikelihood() {
		double logLikelihood = 0.0;
		//int nonZeroTopics;

		// The likelihood of the model is a combination of a 
		// Dirichlet-multinomial for the words in each topic
		// and a Dirichlet-multinomial for the topics in each
		// document.

		// The likelihood function of a dirichlet multinomial is
		//	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
		//	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )

		// So the log likelihood is 
		//	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
		//	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]

		// Do the documents first

		int[] topicCounts = new int[numTopics];
		double[] topicLogGammas = new double[numTopics];
		int[] docTopics;

		for (int topic=0; topic < numTopics; topic++) {
			topicLogGammas[ topic ] = Dirichlet.logGammaStirling( alpha );
		}

		for (int doc=0; doc < data.size(); doc++) {
			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

			docTopics = topicSequence.getFeatures();

			for (int token=0; token < docTopics.length; token++) {
				topicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				if (topicCounts[topic] > 0) {
					logLikelihood += (Dirichlet.logGammaStirling(alpha + topicCounts[topic]) -
							topicLogGammas[ topic ]);
				}
			}

			// subtract the (count + parameter) sum term
			logLikelihood -= Dirichlet.logGammaStirling(alphaSum + docTopics.length);

			Arrays.fill(topicCounts, 0);
		}

		// add the parameter sum term
		logLikelihood += data.size() * Dirichlet.logGammaStirling(alphaSum);

		// And the topics

		// Count the number of type-topic pairs that are not just (logGamma(beta) - logGamma(beta))
		int nonZeroTypeTopics = 0;

		for (int type=0; type < numTypes; type++) {
			// reuse this array as a pointer

			topicCounts = typeTopicCounts[type];

			for (int topic = 0; topic < numTopics; topic++) {
				if (topicCounts[topic] == 0) { continue; }

				nonZeroTypeTopics++;
				logLikelihood += Dirichlet.logGammaStirling(beta + topicCounts[topic]);

				if (Double.isNaN(logLikelihood)) {
					System.out.println("NaN in log likelihood calculation: " + topicCounts[topic]);
					System.exit(1);
				} 
				else if (Double.isInfinite(logLikelihood)) {
					logger.warning("infinite log likelihood");
					System.exit(1);
				}
			}
		}

		for (int topic=0; topic < numTopics; topic++) {
			logLikelihood -= 
					Dirichlet.logGammaStirling( (beta * numTypes) +
							tokensPerTopic[ topic ] );

			if (Double.isNaN(logLikelihood)) {
				logger.info("NaN after topic " + topic + " " + tokensPerTopic[ topic ]);
				return 0;
			}
			else if (Double.isInfinite(logLikelihood)) {
				logger.info("Infinite value after topic " + topic + " " + tokensPerTopic[ topic ]);
				return 0;
			}

		}

		// logGamma(|V|*beta) for every topic
		logLikelihood += 
				Dirichlet.logGammaStirling(beta * numTypes) * numTopics;

		// logGamma(beta) for all type/topic pairs with non-zero count
		logLikelihood -=
				Dirichlet.logGammaStirling(beta) * nonZeroTypeTopics;

		if (Double.isNaN(logLikelihood)) {
			logger.info("at the end");
		}
		else if (Double.isInfinite(logLikelihood)) {
			logger.info("Infinite value beta " + beta + " * " + numTypes);
			return 0;
		}

		return logLikelihood;
	}

	@Override
	public int[][] getZIndicators() {
		int [][] indicators = new int[data.size()][];
		for (int doc = 0; doc < data.size(); doc++) {
			FeatureSequence tokenSequence =	(FeatureSequence) data.get(doc).instance.getData();
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			int[] oneDocTopics = topicSequence.getFeatures();
			int docLength = tokenSequence.getLength();
			indicators[doc] = new int [docLength];
			for (int position = 0; position < docLength; position++) {
				indicators[doc][position] = oneDocTopics[position];
			}
		}
		return indicators;
	}
	
	
	/* 
	 * Set the topic indicators in the object
	 *  
	 * This is of interest to compare starting values. 
	 */
	@Override
	public void setZIndicators(int[][] zIndicators) {
		// Set full N matrix to 0
		for( int topic = 0; topic < numTopics; topic++) {
			for ( int type = 0; type < typeTopicCounts.length; type++ ) {
				typeTopicCounts[type][topic] = 0;
			}
			tokensPerTopic[topic] = 0;
		}
		for (int docCnt = 0; docCnt < data.size(); docCnt++) {
			data.get(docCnt).topicSequence = 
					new LabelSequence(topicAlphabet, zIndicators[docCnt]);
			FeatureSequence tokenSequence =
					(FeatureSequence) data.get(docCnt).instance.getData();
			int [] tokens = tokenSequence.getFeatures();
			for (int pos = 0; pos < zIndicators[docCnt].length; pos++) {
				int type = tokens[pos];
				int topic = zIndicators[docCnt][pos];
				typeTopicCounts[type][topic] += 1;
				tokensPerTopic[topic]++;
			}
		}
	}

	@Override
	public int getNoTopics() {
		return numTopics;
	}
	
	@Override
	public int getCurrentIteration() {
		return currentIteration;
	}

	@Override
	public ArrayList<TopicAssignment> getData() {
		return data;
	}
	
	@Override
	public int[][] getDeltaStatistics() {
		throw new NotImplementedException("Delta statistics is not implemented yet! :(");	
	}
	
	@Override
	public int[] getTopTypeFrequencyIndices() {
		throw new NotImplementedException("Type Frequency Indices is not implemented yet! :(");
	}

	@Override
	public int[] getTypeFrequencies() {
		throw new NotImplementedException("Type Frequencies is not implemented yet! :(");
	}

	@Override
	public double[] getTypeMassCumSum() {
		throw new NotImplementedException("Type Frequency Cum Sum is not implemented yet! :(");
	}

	@Override
	public int getCorpusSize() {
		int corpusWordCount = 0;
		for (int doc = 0; doc < data.size(); doc++) {
			FeatureSequence tokens =
					(FeatureSequence) data.get(doc).instance.getData();
			int docLength = tokens.size();
			corpusWordCount += docLength;
		}
		return corpusWordCount;
	}

	@Override
	public int[][] getDocumentTopicMatrix() {
		int [][] res = new int[data.size()][];
		for (int docIdx = 0; docIdx < data.size(); docIdx++) {
			int[] topicSequence = data.get(docIdx).topicSequence.getFeatures();
			res[docIdx] = new int[numTopics];
			for (int position = 0; position < topicSequence.length; position++) {
				int topicInd = topicSequence[position];
				res[docIdx][topicInd]++;
			}
		}
		return res;
	}

	@Override
	public int[][] getTypeTopicMatrix() {
		int [][] res = new int[typeTopicCounts.length][typeTopicCounts[0].length];
		for (int i = 0; i < res.length; i++) {
			for (int j = 0; j < res[i].length; j++) {
				res[i][j] = typeTopicCounts[i][j];
			}
		}
		return res;
	}
	
	public double [][] getZbar() {
		return ModifiedSimpleLDA.getZbar(data,numTopics);
	}

	public double [][] getThetaEstimate() {
		return ModifiedSimpleLDA.getThetaEstimate(data,numTopics,alpha);
	}

	@Override
	public void preIteration() {
		
	}

	@Override
	public void postIteration() {
		
	}

	@Override
	public void preSample() {
		
	}

	@Override
	public void postSample() {
		
	}

	@Override
	public void postZ() {
		
	}

	@Override
	public void preZ() {
		
	}
	
	@Override
	public LDAConfiguration getConfiguration() {
		return config;
	}

	@Override
	public int getNoTypes() {
		return numTypes;
	}

	@Override
	public void addTestInstances(InstanceList testSet) {
		throw new NotImplementedException();
	}

	@Override
	public double getBeta() {
		return beta;
	}
	
	@Override
	public double[] getAlpha() {
		double [] alphaVect = new double[numTopics];
		for (int i = 0; i < alphaVect.length; i++) {
			alphaVect[i] = alpha / numTopics;
		}
		return alphaVect;
	}

	@Override
	public void abort() {
		abort = true;
	}

	@Override
	public boolean getAbort() {
		return abort;
	}

	@Override
	public InstanceList getDataset() {
		return trainingData;
	}
	
	@Override
	public double[] getLogLikelihood() {
		return null;
	}

	@Override
	public double[] getHeldOutLogLikelihood() {
		return null;
	}


	public void addInstances (InstanceList training) {
		trainingData = training;
		alphabet = training.getDataAlphabet();
		numTypes = alphabet.size();

		betaSum = beta * numTypes;

		typeTopicCounts = new int[numTypes][numTopics];

		for (Instance instance : training) {

			FeatureSequence tokens = (FeatureSequence) instance.getData();
			LabelSequence topicSequence =
					new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < tokens.size(); position++) {

				int topic = random.nextInt(numTopics);
				topics[position] = topic;
				tokensPerTopic[topic]++;

				int type = tokens.getIndexAtPosition(position);
				typeTopicCounts[type][topic]++;
			}

			TopicAssignment t = new TopicAssignment (instance, topicSequence);
			data.add (t);
		}

	}
}
