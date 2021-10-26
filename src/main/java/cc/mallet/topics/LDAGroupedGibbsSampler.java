/**
 * Implements the LDA Grouped Gibbs Sampler 
 * 
 * References 
 * 	  George and Doss (2018). Principled Selection of Hyperparameters in the Latent Dirichlet Allocation Model. JMLR.   
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

public class LDAGroupedGibbsSampler extends UncollapsedParallelLDA implements LDAGibbsSampler, LDASamplerWithPhi {

	private static final long serialVersionUID = 20190529L;
	ParallelDirichlet docDirichletSampler;
	ParallelDirichlet topicDirichletSampler;

	public LDAGroupedGibbsSampler(LDAConfiguration config) {
		super(config);
	}

	/**
	 * Imports the training instances and initializes the LDA model internals.
	 */
	@Override
	public void addInstances(InstanceList training) {
		super.addInstances(training);
		this.numDocuments = training.size();
		this.thetaMatrix = new double[this.numDocuments][this.numTypes];
	}

	/**
	 * Overrides the function in the parent class UncollapsedParallelLDA to
	 * incorporate document theta Dirichlet sampling.
	 * 
	 * @param ctx An LDADocSamplingContext object. See class
	 *            UncollapsedLDADocSamplingContext for more details
	 */
	@Override
	protected LDADocSamplingResult sampleTopicAssignmentsParallel(LDADocSamplingContext ctx) {
		String className = this.getClass().getSimpleName();
		FeatureSequence tokens = ctx.getTokens();
		LabelSequence topics = ctx.getTopics();
		final int docLength = tokens.getLength();
		if (docLength == 0)
			return new LDADocSamplingResultDense(new int[0]);

		int[] tokenSequence = tokens.getFeatures();
		int[] oneDocTopics = topics.getFeatures();
		int[] localTopicCounts = new int[numTopics];

		// Find the non-zero words and topic counts that we have in this document
		for (int position = 0; position < docLength; position++) {
			int topicInd = oneDocTopics[position];
			localTopicCounts[topicInd]++;
		}

		// Samples theta for the selected document
		double[] thetaParameter = new double[numTopics];
		for (int topic = 0; topic < numTopics; topic++) {
			thetaParameter[topic] = localTopicCounts[topic] + alpha[topic];
		}
		docDirichletSampler = new ParallelDirichlet(thetaParameter);
		double[] theta = docDirichletSampler.nextDistribution();
		this.thetaMatrix[ctx.getDocId()] = theta;

		// Iterate over the words in the document
		double score, sum;
		double[] topicTermScores = new double[numTopics];
		int myBatch = ctx.getMyBatch();
		int type, oldTopic, newTopic;
		for (int position = 0; position < docLength; position++) {
			type = tokenSequence[position];
			oldTopic = oneDocTopics[position];

			localTopicCounts[oldTopic]--;
			if (localTopicCounts[oldTopic] < 0)
				throw new IllegalStateException(className + ": Invalid count! count for topic:" + oldTopic + " is: " + localTopicCounts[oldTopic]);

			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of
			 * times each token is assigned to a certain topic. Called before and after
			 * taking a sample topic assignment z
			 */
			decrement(myBatch, oldTopic, type);

			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;
			for (int topic = 0; topic < numTopics; topic++) {
				score = theta[topic] * phi[topic][type];
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
				throw new IllegalStateException(className + ": Topic sampled is invalid!");
			}

			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			// Propagates the update to the topic-token assignments
			/**
			 * Used to subtract and add 1 to the local structure containing the number of
			 * times each token is assigned to a certain topic. Called before and after
			 * taking a sample topic assignment z
			 */
			increment(myBatch, newTopic, type);
		}
		return new LDADocSamplingResultDense(localTopicCounts);
	}

	/**
	 * Spreads the sampling of phi matrix rows on different threads Creates
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
				for (int phi = 0; phi < phiMatrix[topic].length; phi++) {
					phiMean[topic][phi] += phiMatrix[topic][phi];
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
