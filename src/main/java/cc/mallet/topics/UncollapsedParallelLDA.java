package cc.mallet.topics;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.topics.randomscan.document.BatchBuilderFactory;
import cc.mallet.topics.randomscan.document.DocumentBatchBuilder;
import cc.mallet.topics.randomscan.topic.TopicBatchBuilder;
import cc.mallet.topics.randomscan.topic.TopicBatchBuilderFactory;
import cc.mallet.topics.randomscan.topic.TopicIndexBuilder;
import cc.mallet.topics.randomscan.topic.TopicIndexBuilderFactory;
import cc.mallet.types.Alphabet;
import cc.mallet.types.ConditionalDirichlet;
import cc.mallet.types.Dirichlet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.SparseDirichlet;
import cc.mallet.util.IndexSorter;
import cc.mallet.util.LDAThreadFactory;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.LoggingUtils;
import cc.mallet.util.Stats;
import gnu.trove.TIntIntHashMap;
import gnu.trove.TIntIntProcedure;


public class UncollapsedParallelLDA extends ModifiedSimpleLDA implements LDAGibbsSampler, LDASamplerWithPhi {

	private static final long serialVersionUID = 1L;
	protected static Logger logger = Logger.getLogger(UncollapsedParallelLDA.class.getName());	
	protected static FileHandler fileHandler; 

	protected double[][] phi; // phi[topic][type]
	// This matrix will hold a cumulated sample of phi, when it is retrieved we calculate the mean by dividing with how many phi we have sampled
	protected double[][] phiMean;
	// How many iterations should the Phi burn in period be
	protected int phiBurnIn = 0;
	protected int phiMeanThin = 0;
	protected int noSampledPhi= 0;

	// This matrix keeps the document theta samples
	protected double[][] thetaMatrix; // a D x K matrix (see whether this is redundant)
	protected double[][] theta; // a D x K matrix --- used only for diagnostics 
	protected int numDocuments;
	protected String whichModel;
	
	// protected int thetaBurnIn = 0;
	// protected int thetaThin = 0;
	// protected boolean saveTheta = false;

	DocumentBatchBuilder bb;
	TopicBatchBuilder tbb;

	protected int[] batchIndexes;

	boolean measureTimings = false;

	int [] deltaNInterval;
	String dNOutputFn;
	DataOutputStream deltaOutput;

	BlockingQueue<Integer> samplingResults = new LinkedBlockingQueue<Integer>();
	BlockingQueue<Object> phiSamplings = new LinkedBlockingQueue<Object>();
	TIntIntHashMap [] globalDeltaNUpdates;

	AtomicInteger [][] batchLocalTopicTypeUpdates;
	
	int corpusWordCount = 0;

	// Matrix M of topic-token assignments
	// We keep this since we often want fast access to a whole topic
	protected int [][] topicTypeCountMapping;
	protected Integer	noTopicBatches;
	protected boolean	debug;
	private ForkJoinPool documentSamplerPool;
	protected ExecutorService	phiSamplePool;
	private ExecutorService	topicUpdaters;
	Object [] topicLocks;

	protected TopicIndexBuilder topicIndexBuilder;

	// Used for inefficiency calculations
	protected int [][] topIndices = null;
	protected boolean computeDocTopicDistances = false; // default is false 

	AtomicInteger kdDensities = new AtomicInteger();
	long [] zTimings;
	long [] countTimings;

	SparseDirichlet dirichletSampler;
	protected boolean savePhiMeans = true;
	protected int hyperparameterOptimizationInterval;
	int documentSplitLimit;
	
	File abortFile = new File("abort");

	private static final int RESOURCE_LOG_INTERVAL = 100; // Log resource usage every 100 iterations
	private long totalHeapMemoryUsed = 0;
	private long totalNonHeapMemoryUsed = 0;
	private long totalThreadCount = 0;
	private int iterationCount = 0;
	private AtomicLong totalIpcTime = new AtomicLong(0); // Tracks total IPC time in nanoseconds
	
	public UncollapsedParallelLDA(LDAConfiguration 	config) {
		super(config);

		this.whichModel = config.getScheme(); // to check which algorithm is running
		
		try {
			String allLogFile = config.getLoggingUtil().getLogDir().getAbsolutePath() + "/" + this.whichModel + "-execution-log.txt";
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
		showTopicsInterval = config.getTopicInterval(LDAConfiguration.TOPIC_INTER_DEFAULT);
		computeDocTopicDistances = config.computeDocTopicDistances(LDAConfiguration.COMPUTE_DOC_TOPIC_DISTANCES_DEFAULT);
		documentSamplerPool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
		
		// With job stealing we can only have one global z / counts timing
		zTimings = new long[1];
		countTimings = new long[1];
		noTopicBatches = config.getNoTopicBatches(LDAConfiguration.NO_TOPIC_BATCHES_DEFAULT);
		documentSplitLimit = config.getDocumentSamplerSplitLimit(LDAConfiguration.DOCUMENT_SAMPLER_SPLIT_LIMIT_DEFAULT);

		debug = config.getDebug();
		this.batchIndexes = new int[config.getNoBatches(LDAConfiguration.NO_BATCHES_DEFAULT)];
		for (int bb = 0; bb < config.getNoBatches(LDAConfiguration.NO_BATCHES_DEFAULT); bb++) batchIndexes[bb] = bb;

		startupThreadPools();

		topicLocks = new Object[numTopics];
		for (int i = 0; i < numTopics; i++) {
			topicLocks[i] = new Object();
		}

		measureTimings = config.getMeasureTiming();

		int  [] defaultVal = {-1};
		deltaNInterval = config.getIntArrayProperty("dn_diagnostic_interval",defaultVal);
		if(deltaNInterval.length > 1) {
			dNOutputFn = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() 
					+ "/delta_n").getAbsolutePath();
			dNOutputFn += "/DeltaNs" + "_noDocs_" + data.size() + "_vocab_" 
					+ numTypes + "_iter_" + currentIteration + ".BINARY";
			try {
				deltaOutput = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(dNOutputFn)));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				throw new IllegalArgumentException(e);
			}
		}

		globalDeltaNUpdates = new TIntIntHashMap[numTopics];
		for (int i = 0; i < globalDeltaNUpdates.length; i++) {
			globalDeltaNUpdates[i] = new TIntIntHashMap();
		}
		
		usingSymmetricAlpha = config.useSymmetricAlpha(LDAConfiguration.SYMMETRIC_ALPHA_DEFAULT);

		// plda: Reads configuration file to save Phi matrix (distributions of topics)
		savePhiMeans = config.savePhiMeans(LDAConfiguration.SAVE_PHI_MEAN_DEFAULT);
		phiBurnIn    = (int)(((double) config.getPhiBurnInPercent(LDAConfiguration.PHI_BURN_IN_DEFAULT) / 100)
						             * config.getNoIterations(LDAConfiguration.NO_ITER_DEFAULT)); 
		phiMeanThin  = config.getPhiMeanThin(LDAConfiguration.PHI_THIN_DEFAULT);

		// saveTheta = config.saveTheta(LDAConfiguration.SAVE_THETA_DEFAULT);
		// thetaBurnIn = config.getThetaBurnIn(LDAConfiguration.THETA_BURN_IN_DEFAULT);
		// thetaThin = config.getThetaThin(LDAConfiguration.THETA_THIN_DEFAULT);

		hyperparameterOptimizationInterval = config.getHyperparamOptimInterval(LDAConfiguration.HYPERPARAM_OPTIM_INTERVAL_DEFAULT);
	}
	
	public int[][] getTopIndices() {
		return topIndices;
	}

	@Override 
	public int getCorpusSize() { return corpusWordCount;	}


	@Override
	public int[][] getTypeTopicCounts() { 
		int [][] tTCounts = new int[numTypes][numTopics];
		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < topicTypeCountMapping[topic].length; type++) {
				tTCounts[type][topic] = topicTypeCountMapping[topic][type];
			}
		}
		return tTCounts;	
	}

	public void debugPrintMMatrix() {
		if(debug) {
			printMMatrix();
		}
	}

	public void printMMatrix() {
		int [][] ttCounts = getTypeTopicCounts();
		printMMatrix(ttCounts,"Type Topic Counts:\n");
	}

	public void printMMatrix(int [][] matrix, String heading) {
		StringBuffer res = new StringBuffer();
		res.append(heading + ":\n");
		res.append("Topic:   ");
		for (int topic = 0; topic < matrix[0].length; topic++) {
			res.append(String.format("%02d, ",topic));
		}
		res.append("\n");
		res.append("-----------------------------------------------------------\n");
		for (int topic = 0; topic < matrix.length; topic++) {
			res.append("[" + String.format("%02d",topic) + "=" + alphabet.lookupObject(topic) + "]: ");
			for (int type = 0; type < matrix[topic].length; type++) {
				if(matrix[topic][type]==0) {
					res.append("    ");
				} else {
					res.append(String.format("%02d, ", matrix[topic][type]));
				}
			}
			res.append("\n");
		}
		System.out.println(res);
	}


	public void ensureConsistentTopicTypeCountDelta(AtomicInteger [][] counts, int batch) {
		int sumtotal = 0;
		//int deltacount = 0;
		for (AtomicInteger [] topic : counts ) {
			for (int type = 0; type < topic.length; type++ ) { 
				sumtotal += topic[type].get();
				//if(topic[type]!=0) deltacount++;
			}
		}
		if(sumtotal != 0) {
			//printMMatrix(counts, "Broken Batch:");
			throw new IllegalArgumentException("(Iteration = " + currentIteration + ", Batch = " + batch + ") Delta does not sum to Zero! Sumtotal: " + sumtotal);
		}
	}

	public void printTopicTypeCountDelta(int [][] counts, int batch) {
		int sumtotal = 0;

		for (int [] topic : counts ) {
			for (int type = 0; type < topic.length; type++ ) { 
				sumtotal += topic[type];
			}
		}
		if(sumtotal!=0)
			System.out.println("Batch: " + batch + " Sumtotal:" + sumtotal);
	}


	public void ensureConsistentTopicTypeCounts(int [][] topicTypeCounts, int[][] typeTopicCounts, int[] tokensPerTopic) {
		int sumtotalTypeTopic = 0;
		int [] typeTopicTTCount = new int [numTopics]; 

		int sumtotalTopicType = 0;
		int [] topicTypeTTCount = new int [numTopics]; 
		
		for (int topic = 0; topic < numTopics; topic++ ) {
			for (int type = 0; type < topicTypeCounts[topic].length; type++ ) { 
				{
					int count = topicTypeCounts[topic][type];
					if(count<0) throw new IllegalArgumentException("TopicTypeCounts: Negative topic count! Topic: " 
							+ topic + " has negative count for type: " + type + " count=" + count);
					sumtotalTopicType += count;
					topicTypeTTCount[topic] += count;
				}
				{
					int countTypeTopic = typeTopicCounts[type][topic];
					if(countTypeTopic<0) throw new IllegalArgumentException("TypeTopicCounts: Negative topic count! Topic: " 
							+ topic + " has negative count for type: " + type + " count=" + countTypeTopic);
					sumtotalTypeTopic += countTypeTopic;
					typeTopicTTCount[topic] += countTypeTopic;
				}
			}
		}
		if(sumtotalTypeTopic != corpusWordCount) {
			throw new IllegalArgumentException("TypeTopicCounts does not sum to nr. types! Sumtotal: " + sumtotalTypeTopic + " no.types: " + corpusWordCount);
		}
		if(sumtotalTopicType != corpusWordCount) {
			throw new IllegalArgumentException("TopicTypeCounts does not sum to nr. types! Sumtotal: " + sumtotalTopicType + " no.types: " + corpusWordCount);
		}
		for (int i = 0; i < numTopics; i++) {
			if(tokensPerTopic[i]!=topicTypeTTCount[i]) {
				throw new IllegalArgumentException("topicTypeTTCount[" + i + "] does not match global tokensPerTopic[" + i + "]");
			}
			if(tokensPerTopic[i]!=typeTopicTTCount[i]) {
				throw new IllegalArgumentException("typeTopicTTCount[" + i + "] does not match global tokensPerTopic[" + i + "]");
			}
		}
	}

	public void ensureTTEquals() {
		int sumTTCounts1 = sum(getTypeTopicCounts());
		int sumTTCounts2 = sum(typeTopicCounts);
		int sumTopicTotalCounts = sum(getTopicTotals());

		if(sumTTCounts1 != sumTTCounts2)
			throw new IllegalStateException(currentIteration + ": Type-topic counts does not equals Type-topic count mapping: " + sumTTCounts1 + " != "+ sumTTCounts1);

		if(sumTTCounts1 != sumTopicTotalCounts)
			throw new IllegalStateException(currentIteration + ": Type-topic counts does not equals Topic-total counts: " + sumTTCounts1 + ", "+ sumTTCounts2 + " != " + sumTopicTotalCounts);

	}

	/**
	 * Imports the training instances and initializes the LDA model internals.
	 */
	@Override
	public void addInstances (InstanceList training) {
		trainingData = training;
		alphabet = training.getDataAlphabet();
		targetAlphabet = training.getTargetAlphabet();
		numTypes = alphabet.size();
		typeCounts = new int[numTypes];
		batchLocalTopicTypeUpdates = new AtomicInteger[numTopics][numTypes];
		for (int i = 0; i < batchLocalTopicTypeUpdates.length; i++) {
			for (int j = 0; j < batchLocalTopicTypeUpdates[i].length; j++) {
				batchLocalTopicTypeUpdates[i][j] = new AtomicInteger();
			}
		}
		dirichletSampler = createDirichletSampler();

		// Initializing fields needed to sample phi
		betaSum = beta * numTypes;
		topicTypeCountMapping = new int [numTopics][numTypes];
		// Transpose of the above
		typeTopicCounts       = new int [numTypes][numTopics];

		Map<Integer,Integer> docLenCnts = new java.util.HashMap<Integer, Integer>();
		

		// Looping over the new instances to initialize the topic assignment randomly
		for (Instance instance : training) {
			FeatureSequence tokens = (FeatureSequence) instance.getData();
			int docLength = tokens.size();

			if (docLength > longestDocLength)
				longestDocLength = docLength;
			
			if(docLenCnts.get(docLength) == null) {
				docLenCnts.put(docLength,0); 
			}
			docLenCnts.put(docLength,docLenCnts.get(docLength) + 1);

			corpusWordCount += docLength;
			LabelSequence topicSequence =
					new LabelSequence(topicAlphabet, new int[ docLength ]);

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < docLength; position++) {
				// Sampling a random topic assignment
				int topic = initialDrawTopicIndicator();
				topics[position] = topic;

				int type = tokens.getIndexAtPosition(position);
				typeCounts[type] += 1;
				updateTypeTopicCount(type, topic, 1);
			}

			//debugPrintDoc(data.size(),tokens.getFeatures(),topicSequence.getFeatures());
			TopicAssignment t = new TopicAssignment (instance, topicSequence);
			data.add (t);
		}
		
		for (int type = 0; type < numTypes; type++) {
			if (typeCounts[type] > maxTypeCount) { maxTypeCount = typeCounts[type]; }
		}
		
		// Below structures should only be used if hyperparameter optimization is turned on
		if(config.getHyperparamOptimInterval(LDAConfiguration.HYPERPARAM_OPTIM_INTERVAL_DEFAULT)>0) {
			topicDocCounts = new int[numTopics][longestDocLength + 1];
			
			documentTopicHistogram = new AtomicInteger[numTopics][longestDocLength + 1];
			for (int i = 0; i < documentTopicHistogram.length; i++) {
				for (int j = 0; j < documentTopicHistogram[i].length; j++) {
					documentTopicHistogram[i][j] = new AtomicInteger();
				}
			}
			
			docLengthCounts = new int[longestDocLength + 1];
			for (int i = 0; i < docLengthCounts.length; i++) {
				if(docLenCnts.get(i)!=null)
					docLengthCounts[i] = docLenCnts.get(i);
			}
		}
		
		betaSum = beta * numTypes;

		typeFrequencyIndex = IndexSorter.getSortedIndices(typeCounts);
		typeFrequencyCumSum = calcTypeFrequencyCumSum(typeFrequencyIndex,typeCounts);
		
		// Initialize the distribution of words in topics, phi, to the prior value
		phi = new double[numTopics][numTypes];
		if(savePhiMeans()) {
			phiMean = new double[numTopics][numTypes];
		}
		// Sample up the initial Phi Matrix according to random initialization
		int [] topicIndices = new int[numTopics];
		for (int i = 0; i < numTopics; i++) {
			topicIndices[i] = i;
		}
		initialSamplePhi(topicIndices, phi);

		bb = BatchBuilderFactory.get(config, this);
		bb.calculateBatch();
		tbb = TopicBatchBuilderFactory.get(config, this);
		topicIndexBuilder = TopicIndexBuilderFactory.get(config,this);
	}

	int initialDrawTopicIndicator() {
		return random.nextInt(numTopics);
	}

	/**
	 * This method can only be called from threads working
	 * on separate topics. It is not thread safe if several threads
	 * work on the same topic
	 * 
	 * @param type
	 * @param topic
	 * @param count
	 */
	protected void updateTypeTopicCount(int type, int topic, int count) {
		topicTypeCountMapping[topic][type] += count;
		typeTopicCounts[type][topic] += count;
		tokensPerTopic[topic] += count;
		if(topicTypeCountMapping[topic][type]<0) {
			System.err.println("Emergency print!");
			debugPrintMMatrix();
			throw new IllegalArgumentException("Negative count for topic: " + topic 
					+ "! Count: " + topicTypeCountMapping[topic][type] + " type:" 
					+ alphabet.lookupObject(type) + "(" + type + ") update:" + count);
		}
	}

	protected void moveTopic(int oldTopic, int newTopic, int resetValue) {
		topicTypeCountMapping[newTopic] = topicTypeCountMapping[oldTopic];
		topicTypeCountMapping[oldTopic] = new int[numTypes];
		if(resetValue!=0) {
			Arrays.fill(topicTypeCountMapping[oldTopic], resetValue);
		} 
		for(int type = 0; type < numTypes; type++) {
			typeTopicCounts[type][newTopic] = typeTopicCounts[type][oldTopic];
			typeTopicCounts[type][oldTopic] = resetValue;
			if(topicTypeCountMapping[newTopic][type]<0) {
				System.err.println("Emergency print!");
				debugPrintMMatrix();
				throw new IllegalArgumentException("Negative count for topic: " + newTopic 
						+ "! Count: " + topicTypeCountMapping[newTopic][type] + " type:" 
						+ alphabet.lookupObject(type) + "(" + type + ")");
			}
		}
		int tmpTpT = tokensPerTopic[newTopic];
		tokensPerTopic[newTopic] = tokensPerTopic[oldTopic];
		tokensPerTopic[oldTopic] = tmpTpT;
		
		double [] tmpTopic = phi[newTopic];
		phi[newTopic] = phi[oldTopic];
		phi[oldTopic] = tmpTopic;
	}

	protected void moveTopic(int oldTopic, int newTopic) {
		int [] tmpMapping = topicTypeCountMapping[newTopic]; 
		topicTypeCountMapping[newTopic] = topicTypeCountMapping[oldTopic];
		topicTypeCountMapping[oldTopic] = tmpMapping;
		for(int type = 0; type < numTypes; type++) {
			int tmpVal = typeTopicCounts[type][newTopic];
			typeTopicCounts[type][newTopic] = typeTopicCounts[type][oldTopic];
			typeTopicCounts[type][oldTopic] = tmpVal;
			
			if(topicTypeCountMapping[newTopic][type]<0) {
				System.err.println("Emergency print!");
				debugPrintMMatrix();
				throw new IllegalArgumentException("Negative count for topic: " + newTopic 
						+ "! Count: " + topicTypeCountMapping[newTopic][type] + " type:" 
						+ alphabet.lookupObject(type) + "(" + type + ")");
			}
		}
		int tmpCnt = tokensPerTopic[newTopic];
		tokensPerTopic[newTopic] = tokensPerTopic[oldTopic];
		tokensPerTopic[oldTopic] = tmpCnt;
		
		double [] tmpTopic = phi[newTopic];
		phi[newTopic] = phi[oldTopic];
		phi[oldTopic] = tmpTopic;
	}


	private double[] calcTypeFrequencyCumSum(int[] typeFrequencyIndex,int[] typeCounts) {
		double [] result = new double[typeCounts.length];
		result[0] = ((double)typeCounts[typeFrequencyIndex[0]]) / corpusWordCount;
		for (int i = 1; i < typeFrequencyIndex.length; i++) {
			result[i] = (((double)typeCounts[typeFrequencyIndex[i]]) / corpusWordCount) + result[i-1];
		}
		return result;
	}

	/** Trains the LDA model for the given number of iterations. The training is done by sampling z in parallel, updating
	 * the topic-token counts matrix centrally and resampling phi.
	 * 
	 * @see cc.mallet.topics.ModifiedSimpleLDA#sample(int)
	 */
	@Override
	public void sample (int iterations) throws IOException {
		preSample();

		int [] printFirstNDocs = config.getPrintNDocsInterval();
		int nDocs = config.getPrintNDocs();
		int [] printFirstNTopWords = config.getPrintNTopWordsInterval();
		int nWords = config.getPrintNTopWords();

		int [] defaultVal = {-1};
		int [] output_interval = config.getIntArrayProperty("diagnostic_interval",defaultVal);

		// boolean printPhi = config.getPrintPhi();
		boolean savePhi = config.getSavePhi();
		int startDiagnostic = config.getStartDiagnostic(LDAConfiguration.START_DIAG_DEFAULT);
		
		File binOutput = null;
		if(output_interval.length>1||printFirstNDocs.length>1||printFirstNTopWords.length>1) {
			binOutput = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/binaries");
		}
		// --- plda: added on July 10, 2021 --- 
		File asciiOutput = null;
		if (output_interval.length > 1 || printFirstNDocs.length > 1 || printFirstNTopWords.length > 1 || startDiagnostic > 1) {
			asciiOutput = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() + "/ascii");
		}

		long maxExecTimeMillis = TimeUnit.MILLISECONDS.convert(config.getMaxExecTimeSeconds(LDAConfiguration.EXEC_TIME_DEFAULT), TimeUnit.SECONDS); 
		// ----------- plda --------------


		String loggingPath = config.getLoggingUtil().getLogDir().getAbsolutePath();

		boolean computeLikelihood = config.computeLikelihood();
		double logLik; 
		String tw; 
		LogState logState;
		if (computeLikelihood) {
			logLik = modelLogLikelihood();
			tw = topWords(wordsPerTopic);
			logState = new LogState(logLik, 0, tw, loggingPath, logger);
			loglikelihood.add(logLik);
			LDAUtils.logLikelihoodToFile(logState);
		}

		boolean logTypeTopicDensity = config.logTypeTopicDensity(LDAConfiguration.LOG_TYPE_TOPIC_DENSITY_DEFAULT);
		boolean logDocumentDensity = config.logDocumentDensity(LDAConfiguration.LOG_DOCUMENT_DENSITY_DEFAULT);
		boolean logPhiDensity = config.logPhiDensity(LDAConfiguration.LOG_PHI_DENSITY_DEFAULT);
		boolean logTokensPerTopics = config.logTokensPerTopic(LDAConfiguration.LOG_TOKENS_PER_TOPIC);
		double density;
		double docDensity = -1;
		double phiDensity;
		Stats stats;
	
		MarginalProbEstimatorPlain evaluator = null;
		if(testSet != null) {
			evaluator = new MarginalProbEstimatorPlain(numTopics,
					alpha, alphaSum,
					beta,
					typeTopicCounts, 
					tokensPerTopic);
		}
		
		Double heldOutLL = null;
		
		int numParticles = 100;
		if(logTypeTopicDensity || logDocumentDensity || logPhiDensity) {
			density = logTypeTopicDensity ? LDAUtils.calculateMatrixDensity(typeTopicCounts) : -1;
			docDensity = kdDensities.get() / (double) numTopics / data.size();
			phiDensity = logPhiDensity ? LDAUtils.calculatePhiDensity(phi) : -1;
			
			if(testSet != null) {
				heldOutLL = evaluator.evaluateLeftToRight(testSet, numParticles, null);					
			}
			
			if(testSet!=null) {
				stats = new Stats(0, loggingPath, System.currentTimeMillis(), 0, 0, 
						density, docDensity, zTimings, countTimings,phiDensity,heldOutLL);						
			} else {
				stats = new Stats(0, loggingPath, System.currentTimeMillis(), 0, 0, 
					density, docDensity, zTimings, countTimings,phiDensity);
			} 
			
			LDAUtils.logStatstHeaderToFile(stats);
			LDAUtils.logStatsToFile(stats);
		}
		
		if(config.logTopicIndicators(false)) {
			logTopicIndicators();
			System.out.println("Logged topic indicators for iteration: " + getCurrentIteration());
		}

		long zSamplingTimeCum = 0;
		long phiSamplingTimeCum = 0; 
		long diagnosticTimeCum = 0; 
		for (int iteration = 1; iteration <= iterations && !abort; iteration++) {
			currentIteration = iteration;
			if(hyperparameterOptimizationInterval > 1  && iteration % hyperparameterOptimizationInterval == 0) {
				saveHistStats = true;
			}
			preIteration();

			// Saves timestamp
			long iterationStart = System.currentTimeMillis();
			for (int i = 0; i < zTimings.length; i++) {
				zTimings[i] = iterationStart;
			}

			// Sample z by dividing the corpus in batches
			preZ();
			loopOverBatches();

			long beforeSync = System.currentTimeMillis();
			try {
				updateCounts();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			postZ();
			long endTypeTopicUpdate = System.currentTimeMillis();
			long zSamplingTokenUpdateTime = endTypeTopicUpdate - iterationStart;
			logger.finer("Time for updating type-topic counts: " + 
					(endTypeTopicUpdate - beforeSync) + "ms\t");
			zSamplingTimeCum += zSamplingTokenUpdateTime; 
			
			// In the HDP the numTopics can change after the Z sampling 
			if(testSet != null) {
				evaluator = new MarginalProbEstimatorPlain(numTopics,
						alpha, alphaSum,
						beta,
						typeTopicCounts, 
						tokensPerTopic);
			}

			//long beforeSamplePhi = System.currentTimeMillis();
			prePhi();
			samplePhi();
			postPhi();

			long elapsedMillis = System.currentTimeMillis();
			long phiSamplingTime = elapsedMillis - endTypeTopicUpdate;

			logger.finer("Time for sampling phi: " + phiSamplingTime + "ms\t");
			phiSamplingTimeCum += phiSamplingTime; 

			// Start changes on Jan 14, 2022; July 16, 2022 ---------

			long endSamplingTime = System.currentTimeMillis();

			// Log progress at regular intervals
			logMemoryAndThreadMetrics();
			iterationCount++; // Increment the iteration count for logging purposes 
			if (iteration % RESOURCE_LOG_INTERVAL == 0) 
				logDetailedMetrics(iteration, loggingPath);


			int numDocuments = data.size();
			if (startDiagnostic > 0 && iteration >= startDiagnostic) {

				
				if (!this.whichModel.equals("ggs")) {
					// This is for PCGS/UncollapsedParallelLDA, 
					// augmented theta sampling 
					int[][] docTopicCounts = LDAUtils.getDocumentTopicCounts(data, numTopics, numDocuments);
					theta = LDAUtils.drawDirichlets(docTopicCounts, alpha);
				} else { 
					// When GGS, we do not sample theta matrix again
					theta = new double[numDocuments][numTopics];
					for (int didx = 0; didx < numDocuments; didx++) {
						theta[didx] = this.thetaMatrix[didx];
					}
				}

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

				// Compute Quantity B 
				// the phi matrix is KxV; confirmed on June 6, 2025 
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



			if(output_interval.length == 2 && iteration >= output_interval[0] && iteration <= output_interval[1]) {
				LDAUtils.writeBinaryDoubleMatrix(phi, iteration, numTopics, numTypes, binOutput.getAbsolutePath() + "/phi");
				LDAUtils.writeBinaryIntMatrix(typeTopicCounts, iteration, numTypes, numTopics, binOutput.getAbsolutePath() + "/N");
				LDAUtils.writeBinaryIntMatrix(LDAUtils.getDocumentTopicCounts(data, numTopics), iteration, data.size(), numTopics, binOutput.getAbsolutePath() + "/M");
			}

			logger.finer("\nIteration " + iteration + "\tTotal time: " + elapsedMillis + "ms\t");

			// Compute the log likelihood of the model and save to the log file 
			if (computeLikelihood) {
				
				if(testSet != null) {
					heldOutLL = evaluator.evaluateLeftToRight(testSet, numParticles, null);					
					LDAUtils.heldOutLLToFile(loggingPath, iteration, heldOutLL, logger);
					heldOutLoglikelihood.add(heldOutLL);
				}

				logLik = modelLogLikelihood();	
				tw = topWords (wordsPerTopic);
				logState = new LogState(logLik, iteration, tw, loggingPath, logger);
				loglikelihood.add(logLik);
				LDAUtils.logLikelihoodToFile(logState);

				// Occasionally print more information
				if (showTopicsInterval > 0 && (iteration % showTopicsInterval == 0)){
					// logger.info("<" + iteration + "> Log Likelihood: " + logLik);
					logger.fine(tw);
					if(logTypeTopicDensity || logDocumentDensity) {
						density = logTypeTopicDensity ? LDAUtils.calculateMatrixDensity(typeTopicCounts) : -1;
						docDensity = kdDensities.get() / (double) numTopics / data.size();
						phiDensity = logPhiDensity ? LDAUtils.calculatePhiDensity(phi) : -1;
						if(testSet!=null) {
							stats = new Stats(iteration, loggingPath, elapsedMillis, zSamplingTokenUpdateTime, phiSamplingTime, 
									density, docDensity, zTimings, countTimings,phiDensity,heldOutLL);						
						} else {
							stats = new Stats(iteration, loggingPath, elapsedMillis, zSamplingTokenUpdateTime, phiSamplingTime, 
								density, docDensity, zTimings, countTimings,phiDensity);
						}
						LDAUtils.logStatsToFile(stats);
					}
					
					// WARNING: This will SUBSTANTIALLY slow down the sampler
					if(config.logTopicIndicators(false)) {
						logTopicIndicators();
						System.out.println("Logged topic indicators for iteration: " + getCurrentIteration());
					}
					
					if(logTokensPerTopics) {
						LDAUtils.writeIntRowArray(tokensPerTopic, loggingPath +  "/tokens_per_topic.csv");
					}
				}
				
			}

			if( printFirstNTopWords.length > 1 && LDAUtils.inRangeInterval(iteration, printFirstNTopWords)) {
				// Assign these once
				if(topIndices==null) {
					topIndices = LDAUtils.getTopWordIndices(nWords, numTypes, numTopics, typeTopicCounts, alphabet);
				}
				LDAUtils.writeBinaryDoubleMatrixIndices(phi, iteration, binOutput.getAbsolutePath() + "/Selected_Phi_KxV", topIndices);
			}
			
			if( hyperparameterOptimizationInterval > 1 && iteration % hyperparameterOptimizationInterval == 0) {
				optimizeAlpha();
				optimizeBeta();
				
				// Reset counts
				for (int i = 0; i < documentTopicHistogram.length; i++) {
					for (int j = 0; j < documentTopicHistogram[i].length; j++) {
						documentTopicHistogram[i][j].set(0);
					}
				}
				saveHistStats = false;
			}

			kdDensities.set(0);

			postIteration();

			if(abortFile.exists()) {
				abort();
			}

			diagnosticTimeCum += (System.currentTimeMillis() - endSamplingTime);

			if(currentIteration % 100 == 0) { // this is for diagnostics 
				System.out.println("Iteration: " + currentIteration + 
				", Document sampling time: " + zSamplingTimeCum + 
				", Topic sampling time: " + phiSamplingTimeCum + 
				", Total sampling time: " + (zSamplingTimeCum + phiSamplingTimeCum) + 
				", Total diagnostics time: " + diagnosticTimeCum + 
				" (in milliseconds)"); 
			}

			//long iterEnd = System.currentTimeMillis();
			//System.out.println("Iteration "+ currentIteration + " took: " + (iterEnd-iterStart) + " milliseconds...");

			if ((zSamplingTimeCum + phiSamplingTimeCum) >= maxExecTimeMillis){
				break; 
			}
				 
		}
		System.out.println("\nDocument sampling time (millseconds) = " + zSamplingTimeCum);
		System.out.println("\nTopic sampling time (millseconds) = " + phiSamplingTimeCum);
		System.out.println(
			"\nTotal sampling time (millseconds) = " + 
			(zSamplingTimeCum + phiSamplingTimeCum) + 
			" (iteration: " + 
			currentIteration + 
			") \n"
			);
		logAverageMetrics(); // log the average metrics for the entire run
		postSample();
		fileHandler.flush();
	}

	protected void logTopicIndicators() {
		File ld = config.getLoggingUtil().getLogDir();
		File z_file = new File(ld.getAbsolutePath() + "/z_" + getCurrentIteration() + ".csv");
		try (FileWriter fw = new FileWriter(z_file, false); 
				BufferedWriter bw = new BufferedWriter(fw);
				PrintWriter pw  = new PrintWriter(bw)) {

			for (int docIdx = 0; docIdx < data.size(); docIdx++) {
				String szs = "";
				LabelSequence topicSequence =
						(LabelSequence) data.get(docIdx).topicSequence;
				int [] oneDocTopics = topicSequence.getFeatures();
				for (int i = 0; i < topicSequence.size(); i++) {
					szs += oneDocTopics[i] + ",";
				}
				if(szs.length()>0) {
					szs = szs.substring(0, szs.length()-1);
				}
				pw.println(szs);			
			}			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * This method only samples Zbar given Phi, i.e it does not sample/update Phi
	 * 
	 * @param iterations
	 */
	public void sampleZGivenPhi(int iterations) {
		preSample();

		for (int iteration = 1; iteration <= iterations && !abort; iteration++) {
			preIterationGivenPhi();
			currentIteration = iteration;

			// Sample z by dividing the corpus in batches
			preZ();
			loopOverBatches();

			long beforeSync = System.currentTimeMillis();
			try {
				updateCounts();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			postZ();
			long endTypeTopicUpdate = System.currentTimeMillis();
			logger.finer("\nIteration " + iteration + " Time for updating type-topic counts: " + 
				(endTypeTopicUpdate - beforeSync) + "ms\t");

			
			/*
			// Occasionally print more information
			if (showTopicsInterval > 0 && iteration % showTopicsInterval == 0) {
				double logLik = modelLogLikelihood();	
				String tw  = topWords (wordsPerTopic);
				logger.info("<" + iteration + "> Log Likelihood: " + logLik);
				logger.fine(tw);
			}
			*/

			kdDensities.set(0);

			postIterationGivenPhi();
		}

		postSample();
	}
	
	@Override
	public void prePhi() {
		
	}
	
	@Override
	public void postPhi() {

	}


	@Override
	public void postSample() {
		super.postSample();
		// By now we don't need the thread pools any more
		shutdownThreadPools();
		flushDeltaOut();
	}

	void shutdownThreadPools() {
		documentSamplerPool.shutdown();
		try {
			documentSamplerPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		}
		catch (InterruptedException ex) {}

		phiSamplePool.shutdown();
		try {
			phiSamplePool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		}
		catch (InterruptedException ex) {}

		topicUpdaters.shutdown();
		try {
			topicUpdaters.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		}
		catch (InterruptedException ex) {}
	}

	@Override
	public void preSample() {
		super.preSample();
		int  [] defaultVal = {-1};
		deltaNInterval = config.getIntArrayProperty("dn_diagnostic_interval", defaultVal);
		if(deltaNInterval.length > 1) {
			dNOutputFn = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() 
					+ "/delta_n").getAbsolutePath();
			dNOutputFn += "/DeltaNs" + "_noDocs_" + data.size() + "_vocab_" 
					+ numTypes + "_iter_" + currentIteration + ".BINARY";
			try {
				deltaOutput = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(dNOutputFn)));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				throw new IllegalArgumentException(e);
			}
		}
		startupThreadPools();
	}

	void startupThreadPools() {
		// If we call sample again the thread pool have been shutdown so we create a new one
		if(documentSamplerPool == null || documentSamplerPool.isShutdown()) {
			//documentSamplerPool = Executors.newFixedThreadPool(noBatches, new LDAThreadFactory("DocumentSampler"));
			documentSamplerPool = new ForkJoinPool();
		}
		if(phiSamplePool == null || phiSamplePool.isShutdown()) {
			phiSamplePool = Executors.newFixedThreadPool(noTopicBatches,new LDAThreadFactory("PhiSampler"));
		}
		if(topicUpdaters == null || topicUpdaters.isShutdown()) {
			topicUpdaters = Executors.newFixedThreadPool(2,new LDAThreadFactory("TopicUpdater"));
		}
	}

	/**
	 * Returns if 'iter' is in any of the intervals specified by intervals
	 *
	 * @param iter The iteration to check if interval.
	 * @param intervals An integer array of even length.
	 */
	public boolean iterationInInterval(int iter, int[] intervals) {		
		if(intervals.length == 1) return false;
		if(intervals.length % 2 != 0) throw new IllegalArgumentException();

		for(int i = 0; i < intervals.length / 2; i++){
			if(iter >= intervals[2 * i] && iter <= intervals[2 * i + 1]){
				return true;
			}
		}
		return false;
	}	

	protected void updateCounts() throws InterruptedException {
		// Puts together changes to the type-topic counts matrix, by looping over the collection
		// of updates data structures

		// First empty the updates from previous run
		for (int i = 0; i < globalDeltaNUpdates.length; i++) {
			globalDeltaNUpdates[i].clear();
		}

		if(iterationInInterval(currentIteration, deltaNInterval)) {
			flushDeltaOut();
			dNOutputFn = LoggingUtils.checkCreateAndCreateDir(config.getLoggingUtil().getLogDir().getAbsolutePath() 
					+ "/delta_n").getAbsolutePath();
			dNOutputFn += "/DeltaNs" + "_noDocs_" + data.size() + "_vocab_" 
					+ numTypes + "_iter_" + currentIteration + ".BINARY";
			try {
				deltaOutput = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(dNOutputFn)));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				throw new IllegalArgumentException(e);
			}
		}

		long zFin = System.currentTimeMillis();
		zTimings[0] =  zFin - zTimings[0];
		updateTopics();
		countTimings[0] = System.currentTimeMillis() - zFin;

		if(iterationInInterval(currentIteration, deltaNInterval)) {
			flushDeltaOut();
		}
	}

	void flushDeltaOut() {
		if(deltaOutput!=null) {
			try {
				deltaOutput.flush();
				deltaOutput.close();
			} catch (IOException e) {
			 e.printStackTrace();
			 throw new RuntimeException(e);
			}
		}
	}

	
	/**
	 * 'Call' is executed in parallel, but only one topic is touched per thread
	 * so no two threads will update the same topic
	 *
	 */
	class ParallelTopicUpdater implements Callable<Long> {
		int topic;
		public ParallelTopicUpdater(int topic) {
			this.topic = topic;
		}
		@Override
		public Long call() {
			long updates = 0;
			for (int type = 0; type < numTypes; type++) {	
				if(batchLocalTopicTypeUpdates[topic][type].get()!=0) {
					updateTypeTopicCount(type, topic, batchLocalTopicTypeUpdates[topic][type].getAndSet(0));

					// Update delta statistics
					boolean success = globalDeltaNUpdates[topic].increment(type);
					// We need Trove 3.0!!
					if(!success) {
						globalDeltaNUpdates[topic].put(type, 1);
					}

					updates++;
				}
			}
			return updates;
		}   
	}
	
	/*
	void updateTopicsSerial(int batch) {
		for (int topic = (numTopics-1); topic >= 0; topic--) {
			for (int type = (numTypes-1); type >= 0; type--) {	
				if(batchLocalTopicTypeUpdates[batch][topic][type]!=0) {
					updateTypeTopicCount(type, topic, batchLocalTopicTypeUpdates[batch][topic][type]);
					// Now reset the count
					batchLocalTopicTypeUpdates[batch][topic][type] = 0;		
					// Update delta statistics
					boolean success = globalDeltaNUpdates[topic].increment(type);
					// We need Trove 3.0!!
					if(!success) {
						globalDeltaNUpdates[topic].put(type, 1);
					}
				}
			}
		}
	}*/

	void updateTopics() {
		List<ParallelTopicUpdater> builders = new ArrayList<>();
		for (int topic = 0; topic < numTopics; topic++) {
			builders.add(new ParallelTopicUpdater(topic));
		}
		List<Future<Long>> results;
		try {
			results = topicUpdaters.invokeAll(builders);
			for (Future<Long> result : results) {
				result.get();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(-1);
		} catch (ExecutionException e) {
		 e.printStackTrace();
		 System.exit(-1);
		}
	}

	void ensureConsistentPhi(double [][] Phi) {
		for (int i = 0; i < Phi.length; i++) {
			double  sum = 0.0;
			for (int j = 0; j < Phi[i].length; j++) {
				sum += Phi[i][j];
			}
			if(sum>1.01&&sum<0.09&&sum>0) throw new IllegalArgumentException("Inconsistent Phi!");
		}
	}

	/**
	 * Spreads the sampling of phi (topic) matrix rows on different threads. 
	 * This depends on the configuration parameter 'topic_batches'
	 * Creates Runnable() objects that call functions from the superclass
	 * 
	 * TODO: Should be cleaned up!
	 */
	protected void samplePhi() {
		tbb.calculateBatch();
		int[][] topicBatches = tbb.topicBatches();

		for (final int [] topicIndices : topicBatches) {
			final int [][] topicTypeIndices = topicIndexBuilder.getTopicTypeIndices();
			Runnable newTask = new Runnable() {
				public void run() {
					try {
						long beforeThreads = System.currentTimeMillis();
						loopOverTopics(topicIndices, topicTypeIndices, phi);
						logger.finer("Time of Thread: " + 
								(System.currentTimeMillis() - beforeThreads) + "ms\t");
						phiSamplings.put(new Object());
					} catch (Exception ex) {
						ex.printStackTrace();
						throw new IllegalStateException(ex);
					}
				}
			};
			phiSamplePool.execute(newTask);
		}
		int phiSamplingsDone = 0;
		while(phiSamplingsDone<topicBatches.length) {
			try {
				phiSamplings.take();
				phiSamplingsDone++;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		if(savePhiMeans() && samplePhiThisIteration()) {
			noSampledPhi++;
		}
	}

	/**
	 * Samples rows of the phi matrix using the internal data structure for
	 * token-topic assignments
	 * 
	 * WARNING: Assumes that the sufficient statistic, the type-topic counts,
	 * are properly initialized  
	 * 
	 * @param	first	Index of the first row that should be generated
	 * @param	size	Amount of rows to generate
	 * @param	phiMatrix	Pointer to the phi matrix
	 */
	public void initialSamplePhi(int [] indices, double[][] phiMatrix) {
		for (int topic : indices) {
			int [] relevantTypeTopicCounts = topicTypeCountMapping[topic]; 
			// Generates a standard array to feed to the Dirichlet constructor
			// from the dictionary representation. 
			phiMatrix[topic] = dirichletSampler.nextDistribution(relevantTypeTopicCounts);
		}
	}

	/**
	 * Samples new phi values.  
	 * 
	 * If <code>topicTypeIndices</code> is NOT null it will sample phi conditionally
	 * on the indices in <code>topicTypeIndices</code>
	 * 
	 * @param indices indices of the topics that should be sampled, the other ones are skipped 
	 * @param topicTypeIndices matrix containing the indices of the types that should be sampled (per topic)
	 * @param phiMatrix
	 */
	public void loopOverTopics(int [] indices, int[][] topicTypeIndices, double[][] phiMatrix) {
		long beforeSamplePhi = System.currentTimeMillis();
		for (int topic : indices) {
			int [] relevantTypeTopicCounts = topicTypeCountMapping[topic]; 
			// Generates a standard array to feed to the Dirichlet constructor
			// from the dictionary representation. 
			if (topicTypeIndices == null) {
				// Clint's comment: The hyperparameter beta is not added to the topic counts 
				// 'relevantTypeTopicCounts'. Hence, the MCMC chain is not guranteed to converge 
				// to the correct posterior.
				phiMatrix[topic] = dirichletSampler.nextDistribution(relevantTypeTopicCounts);
			} else {
				double[] dirichletParams = new double[numTypes];
				for (int type = 0; type < numTypes; type++) {
					int thisCount = relevantTypeTopicCounts[type];
					dirichletParams[type] = beta + thisCount; 
				}
				
				int[] typeIndicesToSample = topicTypeIndices[topic];
								
				ConditionalDirichlet dist = new ConditionalDirichlet(dirichletParams);
				double [] newPhi = dist.nextConditionalDistribution(phiMatrix[topic],typeIndicesToSample); 
				
				phiMatrix[topic] = newPhi;
			}
			if(savePhiMeans() && samplePhiThisIteration()) {
				for (int v = 0; v < phiMatrix[topic].length; v++) {
					phiMean[topic][v] += phiMatrix[topic][v];
				}
			}
		}
		long elapsedMillis = System.currentTimeMillis();
		long threadId = Thread.currentThread().getId();

		if(measureTimings) {
			PrintWriter pw = LoggingUtils.checkCreateAndCreateLogPrinter(
					config.getLoggingUtil().getLogDir() + "/timing_data",
					"thr_" + threadId + "_Phi_sampling.txt");
			pw.println(beforeSamplePhi + "," + elapsedMillis);
			pw.flush();
			pw.close();
		}
	}

	boolean samplePhiThisIteration() {
		return phiBurnIn > 0 && currentIteration > phiBurnIn && currentIteration % phiMeanThin  == 0;
	}

	class RecursiveDocumentSampler extends RecursiveAction {
		final static long serialVersionUID = 1L;
		double [][] matrix1;
		double [][] matrix2;
		double [][] resultMatrix;
		int startDoc = -1;
		int endDoc = -1;
		int limit = 1000;
		int myBatch = -1;

		public RecursiveDocumentSampler(int startDoc, int endDoc, int batchId, int ll) {
			this.limit = ll;
			this.startDoc = startDoc;
			this.endDoc = endDoc;
			this.myBatch = batchId;
		}

		@Override
		protected void compute() {
			if ( (endDoc-startDoc) <= limit ) {
				for (int docIdx = startDoc; docIdx < endDoc; docIdx++) {
					FeatureSequence tokenSequence =
							(FeatureSequence) data.get(docIdx).instance.getData();
					LabelSequence topicSequence =
							(LabelSequence) data.get(docIdx).topicSequence;
					int [] docTopicHist = sampleTopicAssignmentsParallel (
							new UncollapsedLDADocSamplingContext(tokenSequence, 
									topicSequence, myBatch, docIdx)).getLocalTopicCounts();
					if(docTopicHist!=null && saveHistStats)
						updateGlobalHistogram(docTopicHist);
				}
			}
			else {
				int range = (endDoc-startDoc);
				int startDoc1 = startDoc;
				int endDoc1 = startDoc + (range / 2);
				int startDoc2 = endDoc1;
				int endDoc2 = endDoc;
				invokeAll(new RecursiveDocumentSampler(startDoc1,endDoc1,myBatch + 1,limit),
						new RecursiveDocumentSampler(startDoc2,endDoc2,myBatch + 2,limit));
			}
		}

		private void updateGlobalHistogram(int[] docTopicHist) {
			for (int topic = 0; topic < docTopicHist.length; topic++) {				
				documentTopicHistogram[topic][(int)docTopicHist[topic]].incrementAndGet();
			}
		}
	}

	/*
	class DocumentSampler extends Thread {
		int [] idxs;
		int myBatch;

		public DocumentSampler(int [] docIndices, int myBatch) {
			this.idxs = docIndices;
			this.myBatch = myBatch;
		}

		public void run() {
			try {
				for(int docIdx : idxs) {
					FeatureSequence tokenSequence =
							(FeatureSequence) data.get(docIdx).instance.getData();
					LabelSequence topicSequence =
							(LabelSequence) data.get(docIdx).topicSequence;
					sampleTopicAssignmentsParallel (tokenSequence, topicSequence, myBatch);
				}
			} catch (Exception ex) {
				ex.printStackTrace();
			}
			try {
				samplingResults.put(myBatch);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}*/

	protected void loopOverBatches() {
		RecursiveDocumentSampler dslr = new RecursiveDocumentSampler(0,data.size(),0,documentSplitLimit);                
		documentSamplerPool.invoke(dslr);
	}

	void debugPrintDoc(int doc, int[] tokSeq, int[] topSeq) {
		if(debug) {
			printDoc(doc,tokSeq,topSeq);
		}
	}

	private void printDoc(int doc, int[] tokSeq, int[] topSeq) {
		if(tokSeq==null) { 
			System.out.println("Token Sequence is null");
			return;
		};
		System.out.print("Doc: " + doc + " Tokens:");
		for (int i = 0; i < tokSeq.length; i++) {
			System.out.print(String.format("%02d, ", tokSeq[i]));
		}
		System.out.println();
		if(topSeq==null) { 
			System.out.println("Token Sequence is null");
			return;
		}
		System.out.print("Doc: " + doc + " Topics:");
		for (int i = 0; i < topSeq.length; i++) {
			System.out.print(String.format("%02d, ",topSeq[i]));
		}
		System.out.println();System.out.println();
	}

	protected LDADocSamplingResult sampleTopicAssignmentsParallel(LDADocSamplingContext ctx) {
		FeatureSequence tokens = ctx.getTokens();
		LabelSequence topics = ctx.getTopics();
		int myBatch = ctx.getMyBatch();

		int type, oldTopic, newTopic;

		final int docLength = tokens.getLength();
		if(docLength==0) return new LDADocSamplingResultDense(new int [0]);

		int [] tokenSequence = tokens.getFeatures();
		int [] oneDocTopics = topics.getFeatures();

		int[] localTopicCounts = new int[numTopics];

		// Find the non-zero words and topic counts that we have in this document
		for (int position = 0; position < docLength; position++) {
			int topicInd = oneDocTopics[position];
			localTopicCounts[topicInd]++;
		}

		double score, sum;
		double[] topicTermScores = new double[numTopics];

		//	Iterate over the words in the document
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence[position];
			oldTopic = oneDocTopics[position];
			localTopicCounts[oldTopic]--;
			if(localTopicCounts[oldTopic]<0) 
				throw new IllegalStateException("Counts cannot be negative! Count for topic:" 
						+ oldTopic + " is: " + localTopicCounts[oldTopic]);

			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of times
			 * each token is assigned to a certain topic. Called before and after taking a sample
			 * topic assignment z
			 */
			decrement(myBatch,oldTopic,type);
			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;

			for (int topic = 0; topic < numTopics; topic++) {
				score = (localTopicCounts[topic] + alpha[topic]) * phi[topic][type];
				topicTermScores[topic] = score;
				sum += score;
			}

			// Choose a random point between 0 and the sum of all topic scores
			// The thread local random performs better in concurrent situations 
			// than the standard random which is thread safe and incurs lock 
			// contention
			double U = ThreadLocalRandom.current().nextDouble();
			double sample = U * sum;

			newTopic = -1;
			while (sample > 0.0) {
				newTopic++;
				sample -= topicTermScores[newTopic];
			} 

			// Make sure we actually sampled a valid topic
			if (newTopic < 0 || newTopic >= numTopics) {
				throw new IllegalStateException ("UncollapsedParallelLDA: New valid topic not sampled.");
			}

			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of times
			 * each token is assigned to a certain topic. Called before and after taking a sample
			 * topic assignment z
			 */
			increment(myBatch,newTopic,type);
		}
		return new LDADocSamplingResultDense(localTopicCounts);
	}

	protected void increment(int myBatch, int newTopic, int type) {
		//batchLocalTopicTypeUpdates[myBatch][newTopic][type] += 1;
		batchLocalTopicTypeUpdates[newTopic][type].incrementAndGet();
		//System.out.println("(Batch=" + myBatch + ") Incremented: topic=" + newTopic + " type=" + type + " => " + batchLocalTopicUpdates[myBatch][newTopic][type]);		
	}

	protected void decrement(int myBatch, int oldTopic, int type) {
		//batchLocalTopicTypeUpdates[myBatch][oldTopic][type] -= 1;
		batchLocalTopicTypeUpdates[oldTopic][type].addAndGet(-1);
		//System.out.println("(Batch=" + myBatch + ") Decremented: topic=" + oldTopic + " type=" + type + " => " + batchLocalTopicUpdates[myBatch][oldTopic][type]);
	}

	//	@Override
	//	/* 
	//	 * Uses SimpleLDA logLikelihood calculation
	//	 */
	//	public double modelLogLikelihood() {
	//		// Parent uses typeTopicCounts, fetch these on demand
	//		typeTopicCounts = getTypeTopicCounts();
	//		return super.modelLogLikelihood();
	//	}

	/**
	 * Computes the LDA log posterior (Doss and George, 2025)
	 * Added on May 21, 2025 
	 */
	private double computeLogPosterior() {
		final double EPS = 1e-12; // For numerical stability in log
		double lp = 0.0;
		// int numTopics = phi.length; this is already set in the constructor 
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

	/* 
	 * Uses AD-LDA logLikelihood calculation
	 *  
	 * Here we override SimpleLDA's original likelihood calculation and use the
	 * AD-LDA logLikelihood calculation. 
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
			topicLogGammas[ topic ] = Dirichlet.logGammaStirling( alpha[topic] );
		}

		for (int doc=0; doc < data.size(); doc++) {
			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

			docTopics = topicSequence.getFeatures();

			for (int token=0; token < topicSequence.size(); token++) {
				topicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				if (topicCounts[topic] > 0) {
					logLikelihood += (Dirichlet.logGammaStirling(alpha[topic] + topicCounts[topic]) -
							topicLogGammas[ topic ]);
				}
			}

			// subtract the (count + parameter) sum term

			logLikelihood -= Dirichlet.logGammaStirling(alphaSum + topicSequence.size());
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
				int topicTypeCount = topicCounts[topic];
				if (topicTypeCount == 0) { continue; }
				
				nonZeroTypeTopics++;
				logLikelihood += Dirichlet.logGammaStirling(beta + topicTypeCount);

				if (Double.isNaN(logLikelihood)) {
					System.err.println("NaN in log likelihood calculation: " + topicTypeCount);
					System.exit(1);
				} 
				else if (Double.isInfinite(logLikelihood)) {
					logger.warning("infinite log likelihood");
					System.exit(1);
				}
			}
		}

		for (int topic=0; topic < numTopics; topic++) {
			int tokensPerTopicK = tokensPerTopic[ topic ];
			logLikelihood -= 
					Dirichlet.logGammaStirling( (beta * numTypes) +
							tokensPerTopicK );

			if (Double.isNaN(logLikelihood)) {
				logger.info("NaN after topic " + topic + " " + tokensPerTopicK);
				return 0;
			}
			else if (Double.isInfinite(logLikelihood)) {
				logger.info("Infinite value after topic " + topic + " " + tokensPerTopicK);
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
	/*
	 * This was copied from SimpleLDA and updated to use the new internal representations
	 * 
	 * @see cc.mallet.topics.SimpleLDA#topWords(int)
	 */
	public String topWords (int numWords) {

		StringBuilder output = new StringBuilder();

		IDSorter[] sortedWords = new IDSorter[numTypes];

		for (int topic = 0; topic < numTopics; topic++) {

			int [] typeMap = topicTypeCountMapping[topic];
			for (int token = 0; token < numTypes; token++) {
				Integer thisCount = typeMap[token];
				sortedWords[token] = new IDSorter(token, (thisCount != null) ? thisCount : 0);
			}

			Arrays.sort(sortedWords);

			output.append(topic + "\t" + tokensPerTopic[topic] + "\t");
			for (int i=0; i < numWords; i++) {
				output.append(alphabet.lookupObject(sortedWords[i].getID()) + " ");
			}
			output.append("\n");
		}

		return output.toString();
	}

	@Override
	public void setConfiguration(LDAConfiguration config) {
		super.setConfiguration(config);	
	}

	public void setZIndicators(int[][] zIndicators) {
		// First reset the counts so new counts are not added to old ones
		batchLocalTopicTypeUpdates = new AtomicInteger[numTopics][numTypes];
		for (int i = 0; i < batchLocalTopicTypeUpdates.length; i++) {
			for (int j = 0; j < batchLocalTopicTypeUpdates[i].length; j++) {
				batchLocalTopicTypeUpdates[i][j] = new AtomicInteger();
			}
		}
		for( int topic = 0; topic < numTopics; topic++) {
			for ( int type = 0; type < numTypes; type++ ) {
				topicTypeCountMapping[topic][type] = 0;
				typeTopicCounts[type][topic] = 0;
			}
			tokensPerTopic[topic] = 0;
		}

		int sumtotal = 0;
		for (int docCnt = 0; docCnt < data.size(); docCnt++) {
			data.get(docCnt).topicSequence = 
					new LabelSequence(topicAlphabet, zIndicators[docCnt]);
			FeatureSequence tokenSequence =
					(FeatureSequence) data.get(docCnt).instance.getData();
			int [] tokens = tokenSequence.getFeatures();
			sumtotal += zIndicators[docCnt].length;
			for (int pos = 0; pos < zIndicators[docCnt].length; pos++) {
				int type = tokens[pos];
				int topic = zIndicators[docCnt][pos];
				updateTypeTopicCount(type, topic, 1);
			}
		}

		if(sumtotal != corpusWordCount) {
			throw new IllegalArgumentException("Count does not sum to nr. types! Sumtotal: " + sumtotal + " no.types: " + corpusWordCount);
		}

		if(logger.getLevel()==Level.INFO) {
			System.out.println("loaded sumtotal: " + sumtotal + " tokens");
		}
		
		int [] topicIndices = new int[numTopics];
		for (int i = 0; i < numTopics; i++) {
			topicIndices[i] = i;
		}

		// This call samples phi given the new topic indicators
		initialSamplePhi(topicIndices, phi);
	}

	/**
	 * Return the type indices for non-zero count updates in the last iteration
	 *
	 */
	@Override
	public int[][] getDeltaStatistics() {
		int [][] topicTypeUpdates = new int[numTopics][];
		for (int topic = 0; topic < topicTypeUpdates.length; topic++) {
			topicTypeUpdates[topic] = new int[globalDeltaNUpdates[topic].size()];
			TIntIntHashMap topicUpdates = globalDeltaNUpdates[topic];
			// Remove Zero count updates
			topicUpdates.retainEntries(new TIntIntProcedure() {
				@Override
				public boolean execute(int a, int b) {
					return b>0;
				}
			});
			// now we can get the keys which are the non zero type indices for this topic
			topicTypeUpdates[topic] = topicUpdates.keys();
		}
		return topicTypeUpdates;
	}

	/**
	 * This is not used yet, current random scan only looks at most frequent words
	 */
	class TypeChangePair implements Comparable<TypeChangePair> {
		public int type;
		public int deltaCount = 0;
		public TypeChangePair(int type, int deltaCount) {
			super();
			this.type = type;
			this.deltaCount = deltaCount;
		}
		@Override
		public int compareTo(TypeChangePair o) {
			return deltaCount - o.getDeltaCount();
		}
		public int getType() {
			return type;
		}
		public void setType(int type) {
			this.type = type;
		}
		public int getDeltaCount() {
			return deltaCount;
		}
		public void setDeltaCount(int deltaCount) {
			this.deltaCount = deltaCount;
		}
	}
	
	public void setPhi(double[][] phi) {
		this.phi = phi;
		if(savePhiMeans()) {
			phiMean = new double[numTopics][numTypes];
		}

	}

	/**
	 * Safer version of setPhi where the data and target alphabets are compared 
	 * to ensure that the vocabularies and document classes are the same as in
	 * the sampler that generated Phi
	 * @param phi
	 * @param dataAlphabet
	 * @param targetAlphabet
	 */
	public void setPhi(double[][] phi, Alphabet dataAlphabet, Alphabet targetAlphabet) {
		if(!dataAlphabet.equals(getAlphabet())) {
			throw new IllegalArgumentException("Vocabularies does not match!");
		}
		if(!targetAlphabet.equals(this.targetAlphabet)) {
			throw new IllegalArgumentException("Document class labels does not match!");
		}
		
		ensureConsistentPhi(phi);
		this.phi = phi;
		if(savePhiMeans()) {
			phiMean = new double[numTopics][numTypes];
		}
	}
	
	protected boolean savePhiMeans() {
		return savePhiMeans;
	}

	// Nothing to do, hooks for subclasses
	public void preIterationGivenPhi() {
		
	}
	
	// Nothing to do, hooks for subclasses
	public void postIterationGivenPhi() {
		
	}

	/* 
	 * Returns the last sampled Phi
	 */
	@Override
	public double[][] getPhi() {
		return phi;
	}

	/* 
	 * Returns the mean 
	 */
	@Override
	public double[][] getPhiMeans() {
		if(noSampledPhi==0) {
			logger.warning("No Phi has yet been sampled! getPhiMeans returns 'null'. Ensure that you have correctly configured 'phi_mean_burnin' and 'phi_mean_thin'");
			return null;
		}
		double [][] result = new double[phiMean.length][phiMean[0].length];
		for (int i = 0; i < phiMean.length; i++) {
			for (int j = 0; j < phiMean[i].length; j++) {
				result[i][j] = phiMean[i][j] / noSampledPhi;
			}
		}
		return result;
	}

	public int getNoSampledPhi() {
		return noSampledPhi;
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
