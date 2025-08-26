/**
 * Implements the LDA Partilly Collapsed Gibbs Sampler
 * 
 * This code fixes an issue with topic sampling (no smoothing using hyperparameter beta); 
 * otherwise, this implementation is similar to UncollapsedParallelLDA.java  
 * 
 * Created on: January 30, 2024 
 * 
 * References 
 * 	  Magnusson et al. (2018)    
 */

package cc.mallet.topics;

import java.io.PrintWriter;
import java.util.concurrent.ThreadLocalRandom;
import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.ParallelDirichlet;
import cc.mallet.util.LoggingUtils;

public class LDAPartiallyCollapsedGibbsSampler extends UncollapsedParallelLDA implements LDAGibbsSampler, LDASamplerWithPhi {

	private static final long serialVersionUID = 20240130L;
	ParallelDirichlet topicDirichletSampler;

	public LDAPartiallyCollapsedGibbsSampler(LDAConfiguration config) {
		super(config);
	}

	/**
	 * Imports the training instances and initializes the LDA model internals.
	 
	@Override
	public void addInstances(InstanceList training) {
		super.addInstances(training);
	}*/

	

	/**
	 * Spreads the sampling of phi matrix rows on different threads creates
	 * Runnable() objects that call functions from the superclass
	 * 
	 */
	protected void samplePhi() {
		tbb.calculateBatch();
		int[][] topicBatches = tbb.topicBatches();

		for (final int[] topicIndices : topicBatches) {
			Runnable newTask = new Runnable() {
				public void run() {
					try {
						long beforeThreads = System.currentTimeMillis();
						loopOverTopics(topicIndices, phi);
						logger.finer("Time of Thread: " + (System.currentTimeMillis() - beforeThreads) + "ms\t");
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
		while (phiSamplingsDone < topicBatches.length) {
			try {
				phiSamplings.take();
				phiSamplingsDone++;
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		if (savePhiMeans() && samplePhiThisIteration()) {
			noSampledPhi++;
		}
	}

	/**
	 * Overloads the function in the parent class UncollapsedParallelLDA to sample
	 * topic Dirichlets using smoothed counts.
	 * 
	 * Samples new topic distributions (phi's).
	 * 
	 * @param indices   indices of the topics to be sampled, the other ones are skipped
	 * @param phiMatrix the K x V topic matrix 
	 */
	protected void loopOverTopics(int[] indices, double[][] phiMatrix) {
		long beforeSamplePhi = System.currentTimeMillis();
		for (int topic : indices) {
			int[] relevantTypeTopicCounts = topicTypeCountMapping[topic]; // to feed to the Dirichlet constructor
			double[] dirichletParams = new double[numTypes];
			for (int type = 0; type < numTypes; type++) {
				dirichletParams[type] = beta + relevantTypeTopicCounts[type];
			}
			topicDirichletSampler = new ParallelDirichlet(dirichletParams);
			phiMatrix[topic] = topicDirichletSampler.nextDistribution();

			if (savePhiMeans() && samplePhiThisIteration()) {
				for (int v = 0; v < phiMatrix[topic].length; v++) {
					phiMean[topic][v] += phiMatrix[topic][v];
				}
			}
		}
		long elapsedMillis = System.currentTimeMillis();
		long threadId = Thread.currentThread().getId();

		if (measureTimings) {
			PrintWriter pw = LoggingUtils.checkCreateAndCreateLogPrinter(
					config.getLoggingUtil().getLogDir() + "/timing_data", "thr_" + threadId + "_Phi_sampling.txt");
			pw.println(beforeSamplePhi + "," + elapsedMillis);
			pw.flush();
			pw.close();
		}
	}
}
