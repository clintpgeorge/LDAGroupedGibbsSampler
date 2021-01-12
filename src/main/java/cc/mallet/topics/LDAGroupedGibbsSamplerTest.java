/**
 * Implements the LDA Grouped Gibbs Sampler. This is not a valid sampler. 
 * 
 * References 
 * 	  1. George and Doss (2015). Principled Selection of Hyperparameters in the Latent Dirichlet Allocation Model. JMLR.   
 */

package cc.mallet.topics;

import java.util.concurrent.ThreadLocalRandom;
import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.ParallelDirichlet;

public class LDAGroupedGibbsSamplerTest extends UncollapsedParallelLDA implements LDAGibbsSampler, LDASamplerWithPhi {

	private static final long serialVersionUID = 20190529L;
	ParallelDirichlet docDirichletSampler;
	ParallelDirichlet topicDirichletSampler;
	protected double[][] thetaMatrix; // a D x K matrix
	protected int numDocuments;

	public LDAGroupedGibbsSamplerTest(LDAConfiguration config) {
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
				throw new IllegalStateException("LDAGroupedGibbsSampler: Counts cannot be negative! Count for topic:"
						+ oldTopic + " is: " + localTopicCounts[oldTopic]);

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
				throw new IllegalStateException("LDAGroupedGibbsSampler: New valid topic not sampled.");
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

}
