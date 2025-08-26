package cc.mallet.topics;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import org.apache.commons.lang.NotImplementedException;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.LoggingUtils;
import cc.mallet.util.Randoms;
import cc.mallet.util.Timing;

public class SerialCollapsedLDA extends SimpleLDA implements LDAGibbsSampler {

	private static final long serialVersionUID = 7533987649605469394L;
	protected static Logger logger = Logger.getLogger(SerialCollapsedLDA.class.getName());
	protected static FileHandler fileHandler; 
	LDAConfiguration config;
	int currentIteration = 0 ;
	private int startSeed;
	boolean abort = false;

	private static final int RESOURCE_LOG_INTERVAL = 100; // Log resource usage every 100 iterations
	private long totalHeapMemoryUsed = 0;
	private long totalNonHeapMemoryUsed = 0;
	private long totalThreadCount = 0;
	private int iterationCount = 0;
	private AtomicLong totalIpcTime = new AtomicLong(0); // Tracks total IPC time in nanoseconds

	protected double[][] theta;
	protected double[][] phi;
	protected boolean computeDocTopicDistances = false; // default is false 

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

		try {
			String allLogFile = config.getLoggingUtil().getLogDir().getAbsolutePath() + "/" + config.getScheme() + "-execution-log.txt";
			fileHandler = new FileHandler(allLogFile, true);
			fileHandler.setEncoding("UTF-8");
			fileHandler.setLevel(java.util.logging.Level.ALL);
			fileHandler.setFormatter(new SimpleFormatter());
			fileHandler.setFilter(null);
			logger.setLevel(java.util.logging.Level.ALL);
			logger.setUseParentHandlers(false);
			logger.addHandler(fileHandler);
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to initialize FileHandler for logging", e);
		}

		printLogLikelihood = false;
		showTopicsInterval = config.getTopicInterval(LDAConfiguration.TOPIC_INTER_DEFAULT);
		computeDocTopicDistances = config.computeDocTopicDistances(LDAConfiguration.COMPUTE_DOC_TOPIC_DISTANCES_DEFAULT);
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
		long diagnosticTimeCum = 0; 

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

			long endSamplingTime = System.currentTimeMillis();

			// Log progress at regular intervals
			logMemoryAndThreadMetrics();
			iterationCount++; // Increment the iteration count for logging purposes 
			if (iteration % RESOURCE_LOG_INTERVAL == 0) 
				logDetailedMetrics(iteration, loggingPath);

			if(config!= null) { 
				config.getLoggingUtil().logTiming(new Timing(iterationStart,elapsedMillis,"CollapsedSample_Z"));
			}

			if(output_interval.length == 2 && iteration >= output_interval[0] && iteration <= output_interval[1]) {
				LDAUtils.writeBinaryIntMatrix(typeTopicCounts, iteration, numTypes, numTopics, binOutput.getAbsolutePath() + "/Serial_N");
				LDAUtils.writeBinaryIntMatrix(LDAUtils.getDocumentTopicCounts(data, numTopics), iteration, data.size(), numTopics, binOutput.getAbsolutePath() + "/Serial_M");
			}

			if (computeLikelihood) {
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
				double [][] phiM = LDAUtils.drawDirichlets(typeTopicCounts);
				LDAUtils.writeBinaryDoubleMatrixIndices(LDAUtils.transpose(phiM), iteration, binOutput.getAbsolutePath() + "/Selected_Phi_KxV", topIndices);
			}

			// Start changes on Jan 14, 2022 ---------
			int numDocuments = data.size();
			if (startDiagnostic > 0 && iteration >= startDiagnostic) {

				// Augmented Gibbs sampling of \theta
				int[][] docTopicCounts = LDAUtils.getDocumentTopicCounts(data, numTopics, numDocuments);
				theta = LDAUtils.drawDirichlets(docTopicCounts, alpha);

				// Compute Quantiy C_s
				if (computeDocTopicDistances) {
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
				// the typeTopicCounts matrix is VxK; confirmed on June 6, 2025 --- TODO: Need to study this more
				phi = LDAUtils.drawDirichlets(LDAUtils.transpose(typeTopicCounts), beta);

				// Compute Quantity B 
				int K = phi.length; // number of topics
				int V = phi[0].length; // number of unique words in the vocabulary
				if (computeDocTopicDistances) {
					double minTopicDist = 1e+20;
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

				// Start changes on May 21, 2025 ----------
				double logPosterior = computeLogPosterior();	
				LDAUtils.logPosteriorToFile(logPosterior, iteration, loggingPath, logger);
				// End changes on May 21, 2025 ---------
			}
			// End of changes on Jan 14, 2022 ---------

			diagnosticTimeCum += (System.currentTimeMillis() - endSamplingTime);

			if(currentIteration % 100 == 0) { // this is for diagnostics 
				System.out.println("Iteration: " + currentIteration +  
				", z sampling time: " + zSamplingTime + 
				", Total diagnostics time: " + diagnosticTimeCum + 
				" (in milliseconds)"); 
			}


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
		
		logAverageMetrics();
		fileHandler.flush();

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
	/**
	 * Computes the LDA log posterior (Doss and George, 2025)
	 * Added on May 21, 2025 
	 */
	private double computeLogPosterior() {
		final double EPS = 1e-12; // For numerical stability in log
		double lp = 0.0;
		// int numTopics = phi.length; no need to redefine this, already defined in SimpleLD
		int vocabSize = phi[0].length;
		int numDocs = data.size();

		// Precompute log(theta) and log(phi)
		double[][] logTheta = new double[numTopics][numDocs]; // K x D
		double[][] logPhi = new double[numTopics][vocabSize]; // K x V

		for (int k = 0; k < numTopics; k++) {
			for (int d = 0; d < numDocs; d++) {
				logTheta[k][d] = Math.log(theta[d][k] + EPS);
			}
			for (int v = 0; v < vocabSize; v++) {
				logPhi[k][v] = Math.log(phi[k][v] + EPS);
			}
		}

		double[] n_dj = new double[numTopics];
		double[][] m_djt = new double[numTopics][vocabSize];
		for (int d = 0; d < numDocs; d++) {
			Arrays.fill(n_dj, 0.0);
			for (int k = 0; k < numTopics; k++) {
				Arrays.fill(m_djt[k], 0.0);
			}
			FeatureSequence tokenSequence = (FeatureSequence) data.get(d).instance.getData();
			LabelSequence topicSequence = (LabelSequence) data.get(d).topicSequence;
			int[] docTopics = topicSequence.getFeatures();
			for (int position = 0; position < topicSequence.size(); position++) {
				int topic = docTopics[position];
				int type = tokenSequence.getIndexAtPosition(position);
				n_dj[topic] += 1.0;
				m_djt[topic][type] += 1.0;
			}

			// lp += sum over k,v of m_djt[k][v] * logPhi[k][v]
			for (int k = 0; k < numTopics; k++) {
				for (int v = 0; v < vocabSize; v++) {
					double count = m_djt[k][v];
					if (count > 0.0) {
						lp += count * logPhi[k][v];
					}
				}
			}

			// lp += sum over k of (n_dj[k] + alpha - 1) * logTheta[k][d]
			for (int k = 0; k < numTopics; k++) {
				lp += (n_dj[k] + alpha - 1.0) * logTheta[k][d];
			}
		}

		// lp += sum over k,v of (beta - 1) * logPhi[k][v]
		double betaMinus1 = beta - 1.0;
		for (int k = 0; k < numTopics; k++) {
			for (int v = 0; v < vocabSize; v++) {
				lp += betaMinus1 * logPhi[k][v];
			}
		}

		return lp;
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

	private void logMemoryAndThreadMetrics() {
		// Cache MemoryMXBean and ThreadMXBean to avoid repeated lookups
		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();

		// Aggregate memory and thread usage
		totalHeapMemoryUsed += memoryBean.getHeapMemoryUsage().getUsed();
		totalNonHeapMemoryUsed += memoryBean.getNonHeapMemoryUsage().getUsed();
		totalThreadCount += threadBean.getThreadCount();
	}

	private void logDetailedMetrics(int iteration, String loggingPath) {
		String detailsFile = loggingPath + "/log-detail-metrics.txt";

		MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
		MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
		MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();
		ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();

		// Gather metrics
		long heapUsedMB = heapUsage.getUsed() / (1024 * 1024);
		long heapCommittedMB = heapUsage.getCommitted() / (1024 * 1024);
		long heapMaxMB = heapUsage.getMax() / (1024 * 1024);
		long nonHeapUsedMB = nonHeapUsage.getUsed() / (1024 * 1024);
		long nonHeapCommittedMB = nonHeapUsage.getCommitted() / (1024 * 1024);
		long nonHeapMaxMB = nonHeapUsage.getMax() / (1024 * 1024);
		int threadCount = threadBean.getThreadCount();
		int peakThreadCount = threadBean.getPeakThreadCount();
		long totalStartedThreadCount = threadBean.getTotalStartedThreadCount();
		long timestamp = System.currentTimeMillis();

		// Write header if file does not exist
		File file = new File(detailsFile);
		boolean writeHeader = !file.exists();

		try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(file, true)))) {
			if (writeHeader) {
				out.println("Iteration\tTimestamp\tHeapUsedMB\tHeapCommittedMB\tHeapMaxMB\tNonHeapUsedMB\tNonHeapCommittedMB\tNonHeapMaxMB\tThreadCount\tPeakThreadCount\tTotalStartedThreadCount");
			}
			out.printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d%n",
					iteration,
					timestamp,
					heapUsedMB,
					heapCommittedMB,
					heapMaxMB,
					nonHeapUsedMB,
					nonHeapCommittedMB,
					nonHeapMaxMB,
					threadCount,
					peakThreadCount,
					totalStartedThreadCount);
		} catch (IOException e) {
			System.err.println("Failed to write detailed metrics to file: " + e.getMessage());
			throw new IllegalStateException(e);
		}
	}


	/**
	 * Finalize metrics logging after all iterations are completed.
	 */
	private void logAverageMetrics() {
		if (iterationCount > 0) {
			long avgHeapMemoryUsed = totalHeapMemoryUsed / iterationCount;
			long avgNonHeapMemoryUsed = totalNonHeapMemoryUsed / iterationCount;
			long avgThreadCount = totalThreadCount / iterationCount;

			long ipcTimeMillis = totalIpcTime.get() / 1_000_000; // Convert nanoseconds to milliseconds
			long avgIpcTimeMillis = ipcTimeMillis / iterationCount;

			System.out.println(String.format("Average IPC Overhead per Iteration: %d ms", avgIpcTimeMillis));
			System.out.println(String.format("Average Heap Memory Used per Iteration: %d MB", avgHeapMemoryUsed / (1024 * 1024)));
			System.out.println(String.format("Average Non-Heap Memory Used per Iteration: %d MB", avgNonHeapMemoryUsed / (1024 * 1024)));
			System.out.println(String.format("Average Thread Count per Iteration: %d", avgThreadCount));
		}
	}

}
