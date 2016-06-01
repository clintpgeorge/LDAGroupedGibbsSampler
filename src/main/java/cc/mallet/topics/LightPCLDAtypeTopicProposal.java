package cc.mallet.topics;

import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;

import cc.mallet.configuration.LDAConfiguration;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.OptimizedGentleAliasMethodDynamicSize;

/**
 * @author Leif Jonsson
 * 
 * Implementation of the Light-PC-LDA algorithm which uses Metropolis-Hastings to 
 * sample the topic indicators. This algorithm has O(1) to sample one topic
 * indicator no matter how many topics
 * 
 */
public class LightPCLDAtypeTopicProposal extends LightPCLDA {

	private static final long serialVersionUID = 1L;

	double [] topicCountBetaHat = new double[numTopics];
	
	// Sparse matrix structure (Global)
	// Contains a array with nonzero topics as elements per type
	int[][] nonZeroTypeTopics;
	// So we can map back from a topic to where it is in nonZeroTopics vector
	int [][] nonZeroTypeTopicsBackMapping;
	// Sparse global topic counts used to identify positions in nonZeroTypeTopics
	// nonZeroTypeTopicCnt indicates how many non-zero topics there are per type.
	int[] nonZeroTypeTopicCnt;
	// Number of tokens in each type
	int[] tokensPerType;
	
	public LightPCLDAtypeTopicProposal(LDAConfiguration config) {
		super(config);
		tbFactory = new TTTableBuilderFactory();
		
	}
	

	@Override
	public void addInstances (InstanceList training) {
		numTypes = training.getDataAlphabet().size();
		nonZeroTypeTopics = new int[numTypes][numTopics];
		nonZeroTypeTopicsBackMapping = new int[numTypes][numTopics];
		nonZeroTypeTopicCnt = new int[numTypes];
		tokensPerType = new int[numTypes];
		
		super.addInstances(training);
		
		initTokensPerType(); 
		initTopicCountBetaHat();
	}
	
	@Override
	protected void updateTypeTopicCount(int type, int topic, int count) {
		
		if(typeTopicCounts[type][topic] == 0 && count > 0){
			insertNonZeroTopicTypes(topic, type);
		}
		
		super.updateTypeTopicCount(type, topic, count);
		updateTopicCountBetaHat(topic, count);
		
		if(typeTopicCounts[type][topic] == 0 && count < 0){
			removeNonZeroTopicTypes(topic, type);
		}
		
	}
	
	
	class TTTableBuilderFactory implements TableBuilderFactory {
		public Callable<TableBuildResult> instance(int type) {
			return new TTParallelTableBuilder(type);
		}
	}

	class TTParallelTableBuilder implements Callable<TableBuildResult> {
		int type;
		public TTParallelTableBuilder(int type) {
			this.type = type;
		}
		@Override
		public TableBuildResult call() {
			/* Nonsparse solution
			double [] probs = new double[numTopics];
			double typeMass = 0; // Type prior mass
			for (int topic = 0; topic < numTopics; topic++) {
				// TODO: If this works we can use a sparse version instead
				typeMass += probs[topic] = (typeTopicCounts[type][topic] + beta) / topicCountBetaHat[topic];
			}
			*/
			
			double [] probs = new double[nonZeroTypeTopicCnt[type]];
			// Iterate over nonzero topic indicators
			double typeMass = 0.0;
			for (int i = 0; i < nonZeroTypeTopicCnt[type]; i++) {
				typeMass += probs[i] = typeTopicCounts[type][nonZeroTypeTopics[type][i]] / topicCountBetaHat[nonZeroTypeTopics[type][i]];
			}
			
			if(aliasTables[type]==null) {
				int aliasSize = numTopics;
				// TODO Fix so that Alias tables uses a minimum of memory by maximizing the Alias tables size to the number of tokens per type
				aliasTables[type] = new OptimizedGentleAliasMethodDynamicSize(probs, typeMass, aliasSize);
			} else {
				aliasTables[type].reGenerateAliasTable(probs, typeMass);
			}
				
			return new TableBuildResult(type, aliasTables[type], typeMass);
		}   
	}
	
	@Override
	public void preIteration() {
		super.preIteration();
		
	};
	
	protected void initTopicCountBetaHat(){
		for (int topic = 0; topic < numTopics; topic++) {
			topicCountBetaHat[topic] = 0;
			for (int type = 0; type < numTypes; type++) {
				topicCountBetaHat[topic] += typeTopicCounts[type][topic];
			}
			topicCountBetaHat[topic] += betaSum;
		}
	}
	
	protected void updateTopicCountBetaHat(int topic, int count){
		topicCountBetaHat[topic] += count;
	}
	

	@Override
	protected void sampleTopicAssignmentsParallel(LDADocSamplingContext ctx) {
		FeatureSequence tokens = ctx.getTokens();
		LabelSequence topics = ctx.getTopics();
		int myBatch = ctx.getMyBatch();

		final int docLength = tokens.getLength();
		if(docLength==0) return;
		
		int [] tokenSequence = tokens.getFeatures();
		int [] oneDocTopics = topics.getFeatures();

		double[] localTopicCounts = new double[numTopics];
		double[] localTopicCounts_not_i = new double[numTopics];
		
		// Populate topic counts
		int nonZeroTopicCnt = 0; // Only needed for statistics
		for (int position = 0; position < docLength; position++) {
			int topicInd = oneDocTopics[position];
			localTopicCounts[topicInd]++;
			localTopicCounts_not_i[topicInd]++;
			if(localTopicCounts[topicInd]==1) nonZeroTopicCnt++;
		}

		kdDensities.addAndGet(nonZeroTopicCnt);
		
		//	Iterate over the words in the document
		for (int position = 0; position < docLength; position++) {
			int type = tokenSequence[position];
			int oldTopic = oneDocTopics[position]; // z_position
			int newTopic = oldTopic;
			
			if(localTopicCounts[oldTopic]<0) 
				throw new IllegalStateException("LightPC-LDA: Counts cannot be negative! Count for topic:" 
						+ oldTopic + " is: " + localTopicCounts[oldTopic]);

			decrement(myBatch, oldTopic, type);

			// #####################################
			// Word-Topic Proposal 
			// #####################################
			
			// N_{d,Z_i} => # counts of topic Z_i in document d
			
			// Create n_d^{-i}, decrease document topic count with z_i

			localTopicCounts_not_i[oldTopic]--;
			
			double u_w = ThreadLocalRandom.current().nextDouble() * (tokensPerType[type] + beta * numTopics); // (n_w + V*beta) * u where u ~ U(0,1)

			int wordTopicIndicatorProposal = -1;
			if(u_w < tokensPerType[type]) {
				double u = u_w / (double) tokensPerType[type];
				wordTopicIndicatorProposal = nonZeroTypeTopics[type][aliasTables[type].generateSample(u)];
			} else {
				wordTopicIndicatorProposal = (int) (((u_w - tokensPerType[type]) / (beta * numTopics)) * numTopics); // assume symmetric beta, just draws one topic
			}

			// Make sure we actually sampled a valid topic
			if (wordTopicIndicatorProposal < 0 || wordTopicIndicatorProposal > numTopics) {
				throw new IllegalStateException ("Light PC-LDA (Type topic proposal): Sampled invalid topic (" + wordTopicIndicatorProposal + ").");
			}			
			
			if(wordTopicIndicatorProposal!=oldTopic) {
				// If we drew a new topic indicator, do MH step for Word proposal
				double n_d_zi_not_i = localTopicCounts_not_i[oldTopic];
				double n_d_zstar_not_i = localTopicCounts_not_i[wordTopicIndicatorProposal];
				double n_zstar_beta_hat = topicCountBetaHat[wordTopicIndicatorProposal];
				double n_zi_beta_hat = topicCountBetaHat[oldTopic];
				double n_w_zi = typeTopicCounts[type][oldTopic];
				double n_w_zstar = typeTopicCounts[type][wordTopicIndicatorProposal];
								
				double nom = phi[wordTopicIndicatorProposal][type] * (alpha + n_d_zstar_not_i) * (beta + n_w_zi) * n_zstar_beta_hat;
				double denom = phi[oldTopic][type] * (alpha + n_d_zi_not_i)  * (beta + n_w_zstar) * n_zi_beta_hat;
				double pi_w =  nom / denom;
				
				if(pi_w > 1){
					localTopicCounts[oldTopic]--;
					localTopicCounts[wordTopicIndicatorProposal]++;
					oldTopic = wordTopicIndicatorProposal; 
				} else {
					double u_pi_w = ThreadLocalRandom.current().nextDouble();
					boolean accept_pi_w = u_pi_w < pi_w;

					if(accept_pi_w) {				
						localTopicCounts[oldTopic]--;
						localTopicCounts[wordTopicIndicatorProposal]++;
	
						// Set oldTopic to the new wordTopicIndicatorProposal just accepted.
						// By doing this the below document proposal will be relative to the 
						// new best proposal so far
						oldTopic = wordTopicIndicatorProposal;
					} 
				}

			}
			
			// #####################################
			// Document-Topic Proposal  
			// #####################################
			 
			double u_i = ThreadLocalRandom.current().nextDouble() * (oneDocTopics.length + (numTopics*alpha));
			
			int docTopicIndicatorProposal = -1;
			if(u_i < oneDocTopics.length) {
				docTopicIndicatorProposal = oneDocTopics[(int) u_i];
			} else {
				docTopicIndicatorProposal = (int) (((u_i - oneDocTopics.length) / (numTopics*alpha)) * numTopics);
			}
			
			// Make sure we actually sampled a valid topic
			if (docTopicIndicatorProposal < 0 || docTopicIndicatorProposal > numTopics) {
				throw new IllegalStateException ("Light PC-LDA (Type topic proposal): Sampled invalid topic (" + docTopicIndicatorProposal + ").");
			}
			
			if(docTopicIndicatorProposal!=oldTopic) {
				// If we drew a new topic indicator, do MH step for Document proposal
				double n_d_zstar_not_i = localTopicCounts_not_i[docTopicIndicatorProposal];
				double n_d_zi_not_i = localTopicCounts_not_i[oldTopic];
				double n_d_zi = localTopicCounts[oldTopic];
				double n_d_zstar = localTopicCounts[docTopicIndicatorProposal];

				double nom = phi[docTopicIndicatorProposal][type] * (alpha + n_d_zstar_not_i) * (alpha + n_d_zi);
				double denom = phi[oldTopic][type] * (alpha + n_d_zi_not_i) * (alpha + n_d_zstar);
				double pi_d = nom / denom;
				// Calculate MH acceptance Min.(1,ratio) but as an if else
				if (pi_d > 1){
					newTopic = docTopicIndicatorProposal;
				} else {
					double u_pi_d = ThreadLocalRandom.current().nextDouble();
					boolean accept_pi_d = u_pi_d < pi_d;
	
					if (accept_pi_d) {
						newTopic = docTopicIndicatorProposal;
					} else {
						// We did not accept either word or document proposal 
						// so oldTopic is still the best indicator
						newTopic = oldTopic;
					}	
				}
			}
			increment(myBatch, newTopic, type);

			// Make sure we actually sampled a valid topic
			if (newTopic < 0 || newTopic > numTopics) {
				throw new IllegalStateException ("Light PC-LDA (Type topic proposal): Sampled invalid topic (" + newTopic + ").");
			}

			// Remove one count from old topic
			localTopicCounts[oldTopic]--;
			// Update the word topic indicator
			oneDocTopics[position] = newTopic;
			// Put that new topic into the counts
			localTopicCounts[newTopic]++;
			// Make sure the "_i" version is also up to date!
			localTopicCounts_not_i[newTopic]++;
		}
	}	
	
	protected void initTokensPerType() {
		// Initialize tokensPerType
		for (int typeidx = 0; typeidx < numTypes; typeidx++) {
			for (int topicidx = 0; topicidx < numTopics; topicidx++) {
				tokensPerType[typeidx] += typeTopicCounts[typeidx][topicidx];
			}
		}
		// System.out.println("Tokens for type 0: " + tokensPerType[0] + " and BetaSums: " + betaSum);
	}
	
	// TODO: Code is copied from CollapsedLightLDA
	protected synchronized void insertNonZeroTopicTypes(int topic, int type) {
		//// We have a new non-zero topic put it in the last empty and update the others
		nonZeroTypeTopics[type][nonZeroTypeTopicCnt[type]] = topic;
		nonZeroTypeTopicsBackMapping[type][topic] = nonZeroTypeTopicCnt[type];
		nonZeroTypeTopicCnt[type]++;
	}
	
	/*
	 * removeNonZeroTopicTypes() and insertNonZeroTopicTypes() needs to be synchronized
	 * to remove the risk of updating the same type in nonZeroTypeTopicCnt
	 */
	protected synchronized void removeNonZeroTopicTypes(int topic, int type) {
		//// Remove the topic by copying the last element to it
		if (nonZeroTypeTopicCnt[type] < 1) {
			throw new IllegalArgumentException ("CollapsedLightLDA: Cannot remove, count is less than 1 => " + nonZeroTypeTopicCnt[type]);
		}
		int topicIndex = nonZeroTypeTopicsBackMapping[type][topic];
		nonZeroTypeTopicCnt[type]--;
		nonZeroTypeTopics[type][topicIndex] = nonZeroTypeTopics[type][nonZeroTypeTopicCnt[type]];
		nonZeroTypeTopicsBackMapping[type][nonZeroTypeTopics[type][topicIndex]] = topicIndex;
	}
}
