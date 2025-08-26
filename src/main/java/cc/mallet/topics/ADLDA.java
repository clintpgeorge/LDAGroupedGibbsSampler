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
import java.util.Iterator;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import org.apache.commons.lang.NotImplementedException;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.LoggingUtils;
import cc.mallet.util.Randoms;
import cc.mallet.util.Stats;
import cc.mallet.util.Timing;


public class ADLDA extends ParallelTopicModel implements LDAGibbsSampler {

	private static final long serialVersionUID = -3423504261653103647L;
	protected static Logger logger = Logger.getLogger(ADLDA.class.getName());
	protected static FileHandler fileHandler; 	
	transient LDAConfiguration config;
	int currentIteration;
	private int startSeed;
	double [] kdDensities;
	private boolean abort = false;

	private static final int RESOURCE_LOG_INTERVAL = 100; // Log resource usage every 100 iterations
	private long totalHeapMemoryUsed = 0;
	private long totalNonHeapMemoryUsed = 0;
	private long totalThreadCount = 0;
	private int iterationCount = 0;
	private AtomicLong totalIpcTime = new AtomicLong(0); // Tracks total IPC time in nanoseconds

	// The original training data
	InstanceList trainingData;

	protected double[][] phi; 
	protected double[][] theta;
	protected boolean computeDocTopicDistances = false; // default is false 
	
	public ADLDA(LDAConfiguration config) {
		// MALLET uses alphaSum iso. alpha, so we need to multiply with no_topics
		super(config.getNoTopics(LDAConfiguration.NO_TOPICS_DEFAULT), 
				config.getAlpha(LDAConfiguration.ALPHA_DEFAULT)
				*config.getNoTopics(LDAConfiguration.NO_TOPICS_DEFAULT), 
				config.getBeta(LDAConfiguration.BETA_DEFAULT));
		printLogLikelihood = false;
		showTopicsInterval = config.getTopicInterval(LDAConfiguration.TOPIC_INTER_DEFAULT);
		computeDocTopicDistances = config.computeDocTopicDistances(LDAConfiguration.COMPUTE_DOC_TOPIC_DISTANCES_DEFAULT);
		logger.setLevel(Level.INFO);
		setConfiguration(config);
		setOptimizeInterval(0);
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
	}

	@Override
	public void setRandomSeed(int seed) {
		super.setRandomSeed(seed);
		startSeed = seed;
	}

	public int getStartSeed() {
		return startSeed;
	}

	@Override
	public void sample(int iterations) throws IOException {
		numIterations = iterations;
		estimate();
	}

	@Override
	public void estimate () throws IOException {
		if(config==null) throw new IllegalStateException("You must set the configuration before calling 'estimate'");
		String loggingPath = config.getLoggingUtil().getLogDir().getAbsolutePath();
		boolean computeLikelihood = config.computeLikelihood();
		double logLik; 
		String tw; 
		LogState logState;
		if (computeLikelihood) {
			logLik = modelLogLikelihood();
			tw = topWords(wordsPerTopic);
			logState = new LogState(logLik, 0, tw, loggingPath, logger);
			LDAUtils.logLikelihoodToFile(logState);
		}
		
		boolean logTypeTopicDensity = config.logTypeTopicDensity(LDAConfiguration.LOG_TYPE_TOPIC_DENSITY_DEFAULT);
		boolean logDocumentDensity = config.logDocumentDensity(LDAConfiguration.LOG_DOCUMENT_DENSITY_DEFAULT);
		double density;
		double docDensity;
		Stats stats;
		int numThreads = config.getNoBatches(LDAConfiguration.NO_BATCHES_DEFAULT);
		kdDensities = new double[numThreads];

		if(logTypeTopicDensity || logDocumentDensity) {
			density = logTypeTopicDensity ? LDAUtils.calculateMatrixDensity(typeTopicCounts) : -1;
			docDensity = logDocumentDensity ? LDAUtils.calculateDocDensity(kdDensities, numTopics, data.size()) : -1;
			stats = new Stats(0, loggingPath, System.currentTimeMillis(), 0, 0, density, docDensity, null, null,0);
			LDAUtils.logStatstHeaderToFile(stats);
			LDAUtils.logStatsToFile(stats);
		}

		// --- plda: added on July 10, 2021 --- 
		int [] printFirstNDocs = config.getPrintNDocsInterval();
		int nDocs = config.getPrintNDocs();
		int [] printFirstNTopWords = config.getPrintNTopWordsInterval();
		// int nWords = config.getPrintNTopWords();

		int [] defaultVal = {-1};
		int [] output_interval = config.getIntArrayProperty("diagnostic_interval",defaultVal);

		boolean savePhi = config.getSavePhi();
		int startDiagnostic = config.getStartDiagnostic(LDAConfiguration.START_DIAG_DEFAULT);

		File asciiOutput = null;
		if (output_interval.length > 1 || printFirstNDocs.length > 1 || printFirstNTopWords.length > 1) {
			asciiOutput = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/ascii");
		}

		long maxExecTimeMillis = TimeUnit.MILLISECONDS.convert(config.getMaxExecTimeSeconds(LDAConfiguration.EXEC_TIME_DEFAULT), TimeUnit.SECONDS); 

		// ----------- plda --------------

		System.out.print("Document sampling hyperparameter alpha = ");
		for(int w = 0; w < alpha.length; w++){
			System.out.print(alpha[w] + " ");
		}
		System.out.println("\nTopic sampling hyperparameter beta = " + beta + "\n");

		setNumThreads(numThreads);
		long startTime = System.currentTimeMillis();

		MyWorkerRunnable[] runnables = new MyWorkerRunnable[numThreads];

		int docsPerThread = data.size() / numThreads;
		int offset = 0;

		if (numThreads > 1) {

			for (int thread = 0; thread < numThreads; thread++) {
				int[] runnableTotals = new int[numTopics];
				System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);

				int[][] runnableCounts = new int[numTypes][];
				for (int type = 0; type < numTypes; type++) {
					int[] counts = new int[typeTopicCounts[type].length];
					System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
					runnableCounts[type] = counts;
				}

				// some docs may be missing at the end due to integer division
				if (thread == numThreads - 1) {
					docsPerThread = data.size() - offset;
				}

				Randoms random = null;
				if (randomSeed == -1) {
					random = new Randoms();
				}
				else {
					random = new Randoms(randomSeed);
				}

				runnables[thread] = new MyWorkerRunnable(numTopics,
						alpha, alphaSum, beta,
						random, data,
						runnableCounts, runnableTotals,
						offset, docsPerThread);

				runnables[thread].initializeAlphaStatistics(docLengthCounts.length);

				offset += docsPerThread;

			}
		}
		else {

			// If there is only one thread, copy the typeTopicCounts
			//  arrays directly, rather than allocating new memory.

			Randoms random = null;
			if (randomSeed == -1) {
				random = new Randoms();
			}
			else {
				random = new Randoms(randomSeed);
			}

			runnables[0] = new MyWorkerRunnable(numTopics,
					alpha, alphaSum, beta,
					random, data,
					typeTopicCounts, tokensPerTopic,
					offset, docsPerThread);

			runnables[0].initializeAlphaStatistics(docLengthCounts.length);

			// If there is only one thread, we 
			//  can avoid communications overhead.
			// This switch informs the thread not to 
			//  gather statistics for its portion of the data.
			runnables[0].makeOnlyThread();
		}

		ExecutorService executor = Executors.newFixedThreadPool(numThreads);

		long execTimeCumMillis = 0; 
		long diagnosticTimeCum = 0; 

		for (int iteration = 1; iteration <= numIterations && !abort ; iteration++) {
			currentIteration = iteration;

			if (saveStateInterval != 0 && iteration % saveStateInterval == 0) {
				this.printState(new File(stateFilename + '.' + iteration));
			}

			if (saveModelInterval != 0 && iteration % saveModelInterval == 0) {
				this.write(new File(modelFilename + '.' + iteration));
			}

			long iterationStart = System.currentTimeMillis();
			if (numThreads > 1) {

				// Submit runnables to thread pool
				for (int thread = 0; thread < numThreads; thread++) {
					if (iteration > burninPeriod && optimizeInterval != 0 &&
							iteration % saveSampleInterval == 0) {
						runnables[thread].collectAlphaStatistics();
					}

					logger.fine("submitting thread " + thread);
					executor.submit(runnables[thread]);
					//runnables[thread].run();
				}

				// I'm getting some problems that look like 
				//  a thread hasn't started yet when it is first
				//  polled, so it appears to be finished. 
				// This only occurs in very short corpora.
				try {
					Thread.sleep(20);
				} catch (InterruptedException e) {

				}

				boolean finished = false;
				while (! finished) {

					try {
						Thread.sleep(10);
					} catch (InterruptedException e) {

					}

					finished = true;

					// Are all the threads done?
					for (int thread = 0; thread < numThreads; thread++) {
						//logger.info("thread " + thread + " done? " + runnables[thread].isFinished);
						finished = finished && runnables[thread].isFinished();
					}

				}
				long elapsedMillis = System.currentTimeMillis();
				long summingStart = elapsedMillis;
				config.getLoggingUtil().logTiming(new Timing(iterationStart,elapsedMillis,"ADLDASample_Z"));
				sumTypeTopicCounts(runnables);
				long summingEnd = System.currentTimeMillis();
				config.getLoggingUtil().logTiming(new Timing(summingStart,summingEnd,"ADLDASynchronize"));

				for (int thread = 0; thread < numThreads; thread++) {
					int[] runnableTotals = runnables[thread].getTokensPerTopic();
					System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);

					int[][] runnableCounts = runnables[thread].getTypeTopicCounts();
					for (int type = 0; type < numTypes; type++) {
						int[] targetCounts = runnableCounts[type];
						int[] sourceCounts = typeTopicCounts[type];

						int index = 0;
						while (index < sourceCounts.length) {

							if (sourceCounts[index] != 0) {
								targetCounts[index] = sourceCounts[index];
							}
							else if (targetCounts[index] != 0) {
								targetCounts[index] = 0;
							}
							else {
								break;
							}

							index++;
						}
						//System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
					}
				}
				//System.out.println("Z sampling finished");
				//System.out.println("Create next batch took: " + (System.currentTimeMillis() - summingEnd) + " ms");

			}
			else {
				if (iteration > burninPeriod && optimizeInterval != 0 &&
						iteration % saveSampleInterval == 0) {
					runnables[0].collectAlphaStatistics();
				}
				runnables[0].run();
			}

			long elapsedMillis = System.currentTimeMillis() - iterationStart;
			execTimeCumMillis += elapsedMillis; 

			long endSamplingTime = System.currentTimeMillis();

			// Log progress at regular intervals
			logMemoryAndThreadMetrics();
			iterationCount++; // Increment the iteration count for logging purposes 
			if (iteration % RESOURCE_LOG_INTERVAL == 0) 
				logDetailedMetrics(iteration, loggingPath);


			if (computeLikelihood)  {
				logLik = modelLogLikelihood();
				String wt = displayTopWords (wordsPerTopic, false);
				logState = new LogState(logLik, iteration, wt, loggingPath, logger);
				LDAUtils.logLikelihoodToFile(logState);

				if (showTopicsInterval > 0 && (iteration % showTopicsInterval == 0)){
					logger.info("<" + iteration + "> Log Likelihood: " + logLik);
					logger.fine(wt);
					
					if(logTypeTopicDensity || logDocumentDensity) {
						density = logTypeTopicDensity ? LDAUtils.calculateMatrixDensity(typeTopicCounts) : -1;
						for (int i = 0; i < runnables.length; i++) {
							kdDensities[i] = runnables[i].getKdDensity();
						}
						docDensity = logDocumentDensity ? LDAUtils.calculateDocDensity(kdDensities, numTopics, data.size()) : -1;
						stats = new Stats(iteration, loggingPath, System.currentTimeMillis(), elapsedMillis, 0, density, docDensity, null, null,0);
						LDAUtils.logStatsToFile(stats);
					}
				}
				
			}

			if (elapsedMillis < 1000) {
				logger.fine(elapsedMillis + "ms ");
			}
			else {
				logger.fine((elapsedMillis/1000) + "s ");
			}   

			if (iteration > burninPeriod && optimizeInterval != 0 &&
					iteration % optimizeInterval == 0) {

				optimizeAlpha(runnables);
				optimizeBeta(runnables);

				logger.fine("[O " + (System.currentTimeMillis() - iterationStart) + "] ");
			}

//			if (iteration % 10 == 0) {
//				if (printLogLikelihood) {
//					logger.info ("<" + iteration + "> LL/token: " + formatter.format(modelLogLikelihood() / totalTokens));
//				}
//				else {
//					logger.info ("<" + iteration + ">");
//				}
//			}

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

			// Reset densities
			for (int i = 0; i < runnables.length; i++) {
				runnables[i].setKdDensity(0);
			}

			diagnosticTimeCum += (System.currentTimeMillis() - endSamplingTime);

			if(currentIteration % 100 == 0) { // this is for diagnostics 
				System.out.println("Iteration: " + currentIteration +  
				", Execution time: " + execTimeCumMillis + 
				", Total diagnostics time: " + diagnosticTimeCum + 
				" (in milliseconds)"); 
			}

			if (execTimeCumMillis >= maxExecTimeMillis){
				break; 
			}
		}

		executor.shutdownNow();

		long execTimeMillis = System.currentTimeMillis() - startTime; 
		long seconds = Math.round(execTimeMillis / 1000.0);
		long minutes = seconds / 60;	seconds %= 60;
		long hours = minutes / 60;	minutes %= 60;
		long days = hours / 24;	hours %= 24;

		StringBuilder timeReport = new StringBuilder();
		timeReport.append("\nTotal time: ");
		if (days != 0) { timeReport.append(days); timeReport.append(" days "); }
		if (hours != 0) { timeReport.append(hours); timeReport.append(" hours "); }
		if (minutes != 0) { timeReport.append(minutes); timeReport.append(" minutes "); }
		timeReport.append(seconds); timeReport.append(" seconds");

		logger.info(timeReport.toString());
		// logger.info("\nExecution time (milliseconds): " + Long.toString(execTimeMillis));
		System.out.println("\nExecution time (milliseconds): " + Long.toString(execTimeMillis));
		// logger.info(
		// 	"\nExecution time (milliseconds; sampling): " + 
		// 	Long.toString(execTimeCumMillis) + 
		// 	" (iteration: " + 
		// 	currentIteration + 
		// 	") \n");
		
		System.out.println(
			"\nExecution time (milliseconds; sampling): " + 
			Long.toString(execTimeCumMillis) + 
			" (iteration: " + 
			currentIteration + 
			") \n");

		logAverageMetrics(); // log the average metrics for the entire run

		fileHandler.flush();
	}

	/**
	 * Computes the LDA log posterior (Doss and George, 2025)
	 * Added on May 21, 2025 
	 */
	private double computeLogPosterior() {
		final double EPS = 1e-12; // For numerical stability in log
		double lp = 0.0;
		// int numTopics = phi.length;
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

			// lp += sum over k of (n_dj[k] + alpha[k] - 1) * logTheta[k][d]
			for (int k = 0; k < numTopics; k++) {
				lp += (n_dj[k] + alpha[k] - 1.0) * logTheta[k][d];
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

	/**
	 * @return the config
	 */
	public LDAConfiguration getConfig() {
		return config;
	}

	@Override
	public void setConfiguration(LDAConfiguration config) {
		this.config = config;
		showTopicsInterval = config.getTopicInterval(LDAConfiguration.TOPIC_INTER_DEFAULT);
	}

	public int[] getTopicTotals() { return tokensPerTopic; }

	public int[][] getTypeTopicCounts() {
		int[][] ttCounts = new int[numTypes][numTopics];
		for (int type=0; type < numTypes; type++) {
			int [] topicCounts = typeTopicCounts[type];
			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {
				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;
				ttCounts[type][topic] = count;
				index++;
			}
		}
		return ttCounts;
	}

	public String topWords(int noWords) {
		StringBuffer result = new StringBuffer();
		String [][] tws = getTopWords(noWords);
		for (int i = 0; i < tws.length; i++) {
			result.append("Topic " + i + ":");
			for (int j = 0; j < tws[i].length; j++) {
				result.append(tws[i][j] + " ");
			}
			result.append("\n");
		}
		return result.toString();
	}

	public String [][] getTopWords(int noWords) {
		String [][] result = new String[numTopics][noWords];

		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();

		// Print results for each topic
		for (int topic = 0; topic < numTopics; topic++) {
			TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
			int word = 0;
			Iterator<IDSorter> iterator = sortedWords.iterator();

			while (iterator.hasNext() && word < noWords) {
				IDSorter info = iterator.next();
				result[topic][word] = (String) alphabet.lookupObject(info.getID());
				word++;
			}
		}

		return result;
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

	@Override
	public void setZIndicators(int[][] zIndicators) {
		throw new NotImplementedException("Setting start values is not implemented yet! :(");
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
	public double[] getTypeMassCumSum() {
		throw new NotImplementedException("Type Frequency Cum Sum is not implemented yet! :(");
	}

	@Override
	public int[] getTypeFrequencies() {
		throw new NotImplementedException("Type Frequencies is not implemented yet! :(");
	}

	@Override
	public int getCorpusSize() {
		return totalTokens;
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
		return getTypeTopicCounts();
	}
	
	public double [][] getZbar() {
		return ModifiedSimpleLDA.getZbar(data,numTopics);
	}
	
	public double [][] getThetaEstimate() {
		return ModifiedSimpleLDA.getThetaEstimate(data, numTopics, alpha);
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
	public double [] getAlpha() {
		return alpha;
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
		this.trainingData = training;
		super.addInstances(training);
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
