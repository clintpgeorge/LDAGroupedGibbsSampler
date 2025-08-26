package cc.mallet.topics.tui;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.Thread.UncaughtExceptionHandler;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.io.FileNotFoundException;
import java.io.IOException;

import cc.mallet.configuration.ConfigFactory;
import cc.mallet.configuration.Configuration;
import cc.mallet.configuration.LDACommandLineParser;
import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.configuration.ParsedLDAConfiguration;
import cc.mallet.topics.ADLDA;
import cc.mallet.topics.CollapsedLightLDA;
import cc.mallet.topics.EfficientUncollapsedParallelLDA;
import cc.mallet.topics.HDPSamplerWithPhi;
import cc.mallet.topics.LDAGibbsSampler;
import cc.mallet.topics.LDAGroupedGibbsSampler;
import cc.mallet.topics.LDAPartiallyCollapsedGibbsSampler;
import cc.mallet.topics.LDASamplerWithPhi;
import cc.mallet.topics.LightPCLDA;
import cc.mallet.topics.LightPCLDAtypeTopicProposal;
import cc.mallet.topics.NZVSSpaliasUncollapsedParallelLDA;
import cc.mallet.topics.PoissonPolyaUrnHDPLDA;
import cc.mallet.topics.PoissonPolyaUrnHDPLDAInfiniteTopics;
import cc.mallet.topics.PoissonPolyaUrnHLDA;
import cc.mallet.topics.PolyaUrnSpaliasLDA;
import cc.mallet.topics.SerialCollapsedLDA;
import cc.mallet.topics.SpaliasUncollapsedParallelLDA;
import cc.mallet.topics.SpaliasUncollapsedParallelWithPriors;
import cc.mallet.topics.TopicModelDiagnosticsPlain;
import cc.mallet.topics.UncollapsedParallelLDA;
import cc.mallet.types.InstanceList;
import cc.mallet.util.EclipseDetector;
import cc.mallet.util.LDAUtils;
import cc.mallet.util.LoggingUtils;
import cc.mallet.util.TeeStream;
import cc.mallet.util.Timer;

public class ParallelLDA  {
	public static String PROGRAM_NAME = "ParallelLDA";
	public static Thread.UncaughtExceptionHandler exHandler;
	protected static volatile boolean abort = false;
	protected static volatile boolean normalShutdown = false;
	static LDAGibbsSampler plda;
	public static PrintWriter pw;

	private static final String TOP_WORDS_FILENAME = "/TopWords.txt";
	private static final String RELEVANCE_WORDS_FILENAME = "/RelevanceWords.txt";

	public static void main(String[] args) throws Exception {
		ParallelLDA plda = new ParallelLDA();
		plda.doSample(args);
	}
	
	private static LDAGibbsSampler getCurrentSampler() {
		return plda;
	}
		
	public void doSample(String[] args) throws Exception {
        if(args.length == 0) {
            System.err.println("\n" + PROGRAM_NAME + ": No args given, you should typically call it along the lines of: \n" 
                    + "java -cp pLDA-X.X.X.jar cc.mallet.topics.tui.ParallelLDA --run_cfg=src/main/resources/configuration/PLDAConfig.cfg\n" 
                    + "or\n" 
                    + "java -jar pLDA-X.X.X.jar -run_cfg=src/main/resources/configuration/PLDAConfig.cfg\n");
            System.exit(-1);
        }
        
        
        String [] newArgs = EclipseDetector.runningInEclipse(args);
        // If we are not running in eclipse we can install the abort functionality
        if(newArgs==null) {
            final Thread mainThread = Thread.currentThread();
            Runtime.getRuntime().addShutdownHook(new Thread() {
                public void run() {
                    int waitTimeout = 300;
                    if(!normalShutdown) {
                        System.err.println("Running shutdown hook: " + PROGRAM_NAME + " Aborted! Waiting max " 
                    + waitTimeout + "(s) for shutdown...");
                        abort = true;
                        if(getCurrentSampler()!=null) {
                            System.err.println("Calling sampler abort...");
                            getCurrentSampler().abort();
                            try {
                                mainThread.join(waitTimeout * 1000);
                            } catch (InterruptedException e) {
                                System.err.println("Exception during Join..");
                                e.printStackTrace();
                            }
                        }
                    } 
                }
            });
            // Else don't install it, but set args to be the one with "-runningInEclipse" removed
        } else {
            args = newArgs;
        }

        exHandler = buildExceptionHandler();
        
        Thread.setDefaultUncaughtExceptionHandler(exHandler);
        
        System.out.println("We have: " + Runtime.getRuntime().availableProcessors() + " processors available");
        String buildVer = LoggingUtils.getManifestInfo("Implementation-Build", "pLDA");
        String implVer  = LoggingUtils.getManifestInfo("Implementation-Version", "pLDA");
        if(buildVer == null || implVer == null) {
            String gitInfo = LoggingUtils.getLatestCommit();
        	System.out.println("Build/version info not found. GIT info: " + (gitInfo != null ? gitInfo : "N/A"));
        } else {
            System.out.println("Build: " + buildVer + " | Version: " + implVer);
        }
        
        LDACommandLineParser cp = new LDACommandLineParser(args);
        
        // We have to create this temporary config because at this stage if we want to create a new config for each run
        ParsedLDAConfiguration tmpconfig = (ParsedLDAConfiguration) ConfigFactory.getMainConfiguration(cp);			
        
        int numberOfRuns = tmpconfig.getInt("no_runs");
        System.out.println("Doing: " + numberOfRuns + " runs\n");
        // Reading in command line parameters		
        for (int i = 0; i < numberOfRuns; i++) {
            System.out.println("Starting run: " + i);
                        
            LDAConfiguration config = (LDAConfiguration) ConfigFactory.getMainConfiguration(cp);
            LoggingUtils lu = new LoggingUtils();
            String expDir = config.getExperimentOutputDirectory("");
            if(!expDir.equals("")) {
                expDir += "/";
            }
            String logSuitePath = expDir + "RunSuite" + LoggingUtils.getDateStamp();
            System.out.println("Logging to: " + logSuitePath);
            lu.checkAndCreateCurrentLogDir(logSuitePath);
            config.setLoggingUtil(lu);

            String [] configs = config.getSubConfigs();
            for(String conf : configs) {
                lu.checkCreateAndSetSubLogDir(conf);
                config.activateSubconfig(conf);
                int commonSeed = config.getSeed(LDAConfiguration.SEED_DEFAULT);
                
                File lgDir = lu.getLogDir();
                String logFile = lgDir.getAbsolutePath() + "/" + conf + "-console-output.txt";

                try (PrintStream logOut = new PrintStream(new FileOutputStream(logFile, true))) {
                    PrintStream teeStdOut = new TeeStream(System.out, logOut);
                    PrintStream teeStdErr = new TeeStream(System.err, logOut);

                    System.setOut(teeStdOut);
                    System.setErr(teeStdErr);

                    System.out.println("Using Config: " + config.whereAmI());
                    System.out.println("Running subconfig: " + conf);
                    String dataset_fn = config.getDatasetFilename();
                    System.out.println("Number of topics: " + config.getNoTopics(-1));
                    System.out.println("Using dataset: " + dataset_fn);
                    if(config.getTestDatasetFilename()!=null) {
                        System.out.println("Using TEST dataset: " + config.getTestDatasetFilename());
                    }
                    String whichModel = config.getScheme();
                    System.out.println("\nScheme: " + whichModel);

                    InstanceList instances = LDAUtils.loadDataset(config, dataset_fn);
                    instances.getAlphabet().stopGrowth();

                    LDAGibbsSampler model = createModel(config, whichModel);
                    plda = model;
                    
                    model.setRandomSeed(commonSeed);
                    if (config.getTfIdfVocabSize(LDAConfiguration.TF_IDF_VOCAB_SIZE_DEFAULT) > 0) {
                        System.out.println("Top TF-IDF threshold: " + config.getTfIdfVocabSize(LDAConfiguration.TF_IDF_VOCAB_SIZE_DEFAULT));
                    } else {
                        System.out.println("Rare word threshold: " + config.getRareThreshold(LDAConfiguration.RARE_WORD_THRESHOLD));
                    }
                    
                    System.out.println("Vocabulary size: " + instances.getDataAlphabet().size());
                    System.out.println("Number of documents: " + instances.size());
                    System.out.println("Loading data instances...");

                    System.out.println("Config seed: " + config.getSeed(LDAConfiguration.SEED_DEFAULT));
                    System.out.println("Start seed: " + model.getStartSeed());
                    model.addInstances(instances);
                    if(config.getTestDatasetFilename()!=null) {
                        InstanceList testInstances = LDAUtils.loadDataset(config, config.getTestDatasetFilename(),instances.getAlphabet());
                        model.addTestInstances(testInstances);
                    }
                    System.out.println("Loaded " + model.getDataset().size() + " documents, with " + model.getCorpusSize() + " words in total.\n\n");

                    System.out.println("Starting iterations (" + config.getNoIterations(LDAConfiguration.NO_ITER_DEFAULT) + " total).");

                    System.out.println("Starting: " + new Date() + "\n");
                    long startTime = System.nanoTime();
                    Timer t = new Timer();
                    t.start();
                    model.sample(config.getNoIterations(LDAConfiguration.NO_ITER_DEFAULT));
                    t.stop();
                    System.out.println("\nFinished: " + new Date() + "\n");
                    long endTime = System.nanoTime();
                    long totalTime = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
                    System.out.println("Execution time: " + totalTime + " milli seconds \n");
                    

                    int requestedWords = config.getNrTopWords(LDAConfiguration.NO_TOP_WORDS_DEFAULT);
                    
                    if(config.saveDocumentTopicMeans()) {
                        String docTopicMeanFn = config.getDocumentTopicMeansOutputFilename();
                        double [][] means = model.getZbar();
                        LDAUtils.writeASCIIDoubleMatrix(means, lgDir.getAbsolutePath() + "/" + docTopicMeanFn, ",");
                    }
                    
                    if(config.saveDocumentTopicDiagnostics()) {
                        TopicModelDiagnosticsPlain tmd = new TopicModelDiagnosticsPlain(model, requestedWords);
                        String docTopicDiagFn = config.getDocumentTopicDiagnosticsOutputFilename();
                        try (PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + "/" + docTopicDiagFn)) {
                            out.println(tmd.topicsToCsv());
                            out.flush();
                        }
                    }

                    if(config.saveDocumentThetaEstimate()) {
                        String docTopicThetaFn = config.getDocumentTopicThetaOutputFilename();
                        double [][] means = model.getThetaEstimate();
                        LDAUtils.writeASCIIDoubleMatrix(means, lgDir.getAbsolutePath() + "/" + docTopicThetaFn, ",");
                    }

                    if(model instanceof LDASamplerWithPhi) {
                        LDASamplerWithPhi modelWithPhi = (LDASamplerWithPhi) model;
                        if(config.savePhiMeans(LDAConfiguration.SAVE_PHI_MEAN_DEFAULT)) {
                            String docTopicMeanFn = config.getPhiMeansOutputFilename();
                            double [][] means = modelWithPhi.getPhiMeans();
                            if(means!=null) {
                                LDAUtils.writeASCIIDoubleMatrix(means, lgDir.getAbsolutePath() + "/" + docTopicMeanFn, ",");
                            } else {
                                System.err.println("WARNING: ParallelLDA: No Phi means where sampled, not saving Phi means! This is likely due to a combination of configuration settings of phi_mean_burnin, phi_mean_thin and save_phi_mean");
                            }
                            String vocabFn = config.getVocabularyFilename();
                            if(vocabFn==null || vocabFn.length()==0) { vocabFn = "phi_vocabulary.txt"; }
                            String [] vocabulary = LDAUtils.extractVocabulaty(instances.getDataAlphabet());
                            LDAUtils.writeStringArray(vocabulary,lgDir.getAbsolutePath() + "/" + vocabFn);
                        }
                    }

                    saveVocabulary(config, instances, lgDir);
                    saveCorpus(config, instances, lgDir);
                    
                    if(config.saveTermFrequencies(false)) {
                        String termCntFn = config.getTermFrequencyFilename();
                        int [] freqs = LDAUtils.extractTermCounts(instances);
                        LDAUtils.writeIntArray(freqs, lgDir.getAbsolutePath() + "/" + termCntFn);
                    }
                    
                    if(config.saveDocLengths(false)) {
                        String docLensFn = config.getDocLengthsFilename();
                        int [] freqs = LDAUtils.extractDocLength(instances);
                        LDAUtils.writeIntArray(freqs, lgDir.getAbsolutePath() + "/" + docLensFn);
                        
                    }
                    
                    List<String> metadata = new ArrayList<>();
                    metadata.add("No. Topics: " + model.getNoTopics());
                    metadata.add("Start Seed: " + model.getStartSeed());
                    lu.dynamicLogRun(expDir, t, cp, (Configuration) config, null, ParallelLDA.class.getName(),
                            "Convergence", "HEADING", "PLDA", 1, metadata);
                    
                    if(requestedWords>instances.getDataAlphabet().size()) {
                        requestedWords = instances.getDataAlphabet().size();
                    }
                    
                    try (PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + TOP_WORDS_FILENAME)) {
                        out.println(LDAUtils.formatTopWordsAsCsv(
                                LDAUtils.getTopWords(requestedWords, 
                                        model.getAlphabet().size(), 
                                        model.getNoTopics(), 
                                        model.getTypeTopicMatrix(), 
                                        model.getAlphabet())));
                        out.flush();
                    }
                    
                    try (PrintWriter out = new PrintWriter(lgDir.getAbsolutePath() + RELEVANCE_WORDS_FILENAME)) {
                        out.println(LDAUtils.formatTopWordsAsCsv(
                                LDAUtils.getTopRelevanceWords(requestedWords, 
                                        model.getAlphabet().size(), 
                                        model.getNoTopics(), 
                                        model.getTypeTopicMatrix(),  
                                        config.getBeta(LDAConfiguration.BETA_DEFAULT),
                                        config.getLambda(LDAConfiguration.LAMBDA_DEFAULT), 
                                        model.getAlphabet())));
                        out.flush();
                    }
                    
                    if(model instanceof HDPSamplerWithPhi) {
                        printHDPResults(model, lgDir);
                    }
                    
                    System.out.println(new Date() + ": I am done!\n");
                }
            }
            normalShutdown = true;
        }
    
    }

	private void saveVocabulary(LDAConfiguration config, InstanceList instances, File logDir) {
		if (config.saveVocabulary(false)) {
			String vocabFn = config.getVocabularyFilename();
			String[] vocabulary = LDAUtils.extractVocabulaty(instances.getDataAlphabet());
			LDAUtils.writeStringArray(vocabulary, logDir.getAbsolutePath() + "/" + vocabFn);
		}
	}

	private void saveCorpus(LDAConfiguration config, InstanceList instances, File logDir) {
		if (config.saveCorpus(false)) {
			String corpusFn = config.getCorpusFilename();
			int[][] corpus = LDAUtils.extractCorpus(instances);
			try {
				LDAUtils.writeASCIIIntMatrix(corpus, logDir.getAbsolutePath() + "/" + corpusFn, ",");

			} catch (FileNotFoundException e) {
				System.err.println("File not found: " + e.getMessage());
				e.printStackTrace();
			} catch (IOException e) {
				System.err.println("IO Exception: " + e.getMessage());
				e.printStackTrace();
			}
		}
	}

	private UncaughtExceptionHandler buildExceptionHandler() {
        return new Thread.UncaughtExceptionHandler() {
            public void uncaughtException(Thread t, Throwable e) {
                System.err.println(t + " throws exception: " + e);
                if(e instanceof java.io.EOFException) {
                    System.err.println("Ignoring it...");
                } else {
                    e.printStackTrace();
                    if(pw != null) {
                        try {
                            e.printStackTrace(pw);
                            pw.close();
                        } catch (Exception e1) {
                            // Give up!
                        }
                    }
                    System.err.println("Main thread Exiting.");
                    System.exit(-1);
                }
            }
        };
    }

	private void printHDPResults(LDAGibbsSampler model, File lgDir) {
        HDPSamplerWithPhi modelWithPhi = (HDPSamplerWithPhi) model;

        // Topic Occurrence Count
        int[] topicOccurrenceCount = modelWithPhi.getTopicOcurrenceCount();
        System.out.println("Topic Occurrence Count:");
        System.out.println(Arrays.toString(topicOccurrenceCount));
        LDAUtils.writeIntArray(topicOccurrenceCount, lgDir.getAbsolutePath() + "/TopicOccurrenceCount.csv");

        // Active Topics
        List<Integer> activeTopicHistoryList = modelWithPhi.getActiveTopicHistory();
        System.out.println("Active topics:");
        System.out.println(activeTopicHistoryList.toString());
        LDAUtils.writeString(String.join(",", activeTopicHistoryList.toString()), lgDir.getAbsolutePath() + "/ActiveTopics.csv");

        // Active Topics in Data
        List<Integer> activeTopicInDataHistory = modelWithPhi.getActiveTopicInDataHistory();
        System.out.println("Active topics in data:");
        System.out.println(activeTopicInDataHistory.toString());
        LDAUtils.writeString(String.join(",", activeTopicInDataHistory.toString()), lgDir.getAbsolutePath() + "/ActiveTopicsInData.csv");

        // Token Allocation
        int[] tokenAllocation = modelWithPhi.getTopicTotals();
        double tokenSum = Arrays.stream(tokenAllocation).sum();
        double[] tokenAllocationPercent = Arrays.stream(tokenAllocation).mapToDouble(count -> count / tokenSum).toArray();
        double[] tokenAllocationCDF = new double[tokenAllocation.length];

        List<String> percentages = new ArrayList<>();
        List<String> cdfs = new ArrayList<>();
        for (int idx = 0; idx < tokenAllocation.length; idx++) {
            tokenAllocationCDF[idx] = idx == 0 ? tokenAllocationPercent[idx] : tokenAllocationPercent[idx] + tokenAllocationCDF[idx - 1];
            percentages.add(String.format("%.4f", tokenAllocationPercent[idx]));
            cdfs.add(String.format("%.4f", tokenAllocationCDF[idx]));
        }

        System.out.println("Topic Token Allocation Count (sum=" + tokenSum + "):");
        System.out.println(Arrays.toString(tokenAllocation));
        System.out.println("Topic Token Allocation Count (%):");
        System.out.println(percentages.toString());
        System.out.println("Topic Token Allocation CDF (%):");
        System.out.println(cdfs.toString());
    }

    public static LDAGibbsSampler createModel(LDAConfiguration config, String whichModel) {
        LDAGibbsSampler model;
        switch(whichModel) {			
        case "ggs": {
            model = new LDAGroupedGibbsSampler(config);
            System.out.println("LDA Grouped Gibbs Sampler. GGS by George and Doss (2025).");
            break;
        }
        case "adlda": {
            model = new ADLDA(config);
            System.out.println("Approximate Distributed LDA. ADLDA by Newman et al. (2009).");
            break;
        }
        case "pcgs": {
            model = new LDAPartiallyCollapsedGibbsSampler(config);
            System.out.println("Partially Collapsed Gibbs Sampler. PCGS by Magnusson et al. (2018).");
            break;
        }
        case "uncollapsed": {
            model = new UncollapsedParallelLDA(config);
            System.out.println("Uncollapsed Parallell LDA. PCGS by Magnusson et al. (2018).");
            break;
        }
        case "collapsed": {
            model = new SerialCollapsedLDA(config);
            System.out.println("Collapsed Serial LDA. CGS of Griffiths and Steyvers (2004).");
            break;
        }
        case "lightcollapsed": {
            model = new CollapsedLightLDA(config);
            System.out.println("CollapsedLightLDA Parallell LDA.");
            break;
        }
        case "efficient_uncollapsed": {
            model = new EfficientUncollapsedParallelLDA(config);
            System.out.println("EfficientUncollapsedParallelLDA Parallell LDA.");
            break;
        }
        case "spalias": {
            model = new SpaliasUncollapsedParallelLDA(config);
            System.out.println("SpaliasUncollapsed Parallell LDA.");
            break;
        }
        case "polyaurn": {
            model = new PolyaUrnSpaliasLDA(config);
            System.out.println("PolyaUrnSpaliasLDA Parallell LDA.");
            break;
        }
         case "ppu_hlda": {
             model = new PoissonPolyaUrnHLDA(config);
            System.out.println("PoissonPolyaUrnHLDA Parallell HDP.");
             break;
         }
         case "ppu_hdplda": {
             model = new PoissonPolyaUrnHDPLDA(config);
            System.out.println("PoissonPolyaUrnHDPLDA Parallell HDP.");
             break;
         }
         case "ppu_hdplda_all_topics": {
             model = new PoissonPolyaUrnHDPLDAInfiniteTopics(config);
            System.out.println("PoissonPolyaUrnHDPLDAInfiniteTopics Parallell HDP.");
             break;
         }
        case "spalias_priors": {
            model = new SpaliasUncollapsedParallelWithPriors(config);
            System.out.println("SpaliasUncollapsed Parallell LDA with Priors.");
            break;
        }
        case "lightpclda": {
            model = new LightPCLDA(config);
            System.out.println("Light PC LDA.");
            break;
        }
        case "lightpcldaw2": {
            model = new LightPCLDAtypeTopicProposal(config);
            System.out.println("Light PC LDA with proposal 2.");
            break;
        }
        case "nzvsspalias": {
            model = new NZVSSpaliasUncollapsedParallelLDA(config);
            System.out.println("NZVSSpaliasUncollapsedParallelLDA Parallell LDA.");
            break;
        }
        default : {
            System.err.println("Invalid model type. Aborting");
            return null;
        }
        }
        return model;
    }

}
