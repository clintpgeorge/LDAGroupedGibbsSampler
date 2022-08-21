# LDA Gibbs Samplers 

This project is forked from the repository [PartiallyCollapsedLDA](https://github.com/lejon/PartiallyCollapsedLDA). PartiallyCollapsedLDA implemented parallel algorithms mentioned in 
[Magnusson et al. (2018)](#1) and [Terenin et al. (2018)](#2) and for comparative study, a few other Gibbs samplers are available in the literature, e.g., Collapsed Gibbs Sampler ([CGS](#4), Griffiths and Steyvers, 2004), for the Latent Dirichlet Allocation ([LDA](#6)) model. 

The *main* Java `class` is `cc.mallet.topics.tui.ParallelLDA` ([ParallelLDA.java](src/main/java/cc/mallet/topics/tui/ParallelLDA.java)). This fork added/edited the following Gibbs samplers mainly to facilitate analyses in [Doss and George (2022)](#3). 

1. **Collapsed Gibbs Sampler** (CGS, [Griffiths and Steyvers 2004](#4)). It is a serial implementation of CGS; see the class file: [SerialCollapsedLDA.java](src/main/java/cc/mallet/topics/SerialCollapsedLDA.java)
   
2. **Partially Collapsed Gibbs** Sampler (PCGS, [Magnusson et al. 2018](#1)). It is a parallel implementation of PCGS; see the class file: [UncollapsedParallelLDA.java](src/main/java/cc/mallet/topics/UncollapsedParallelLDA.java)

3. **Grouped Gibbs Sampler** (GGS, [Doss and George 2022](#3)). It is a parallel implementation of GGS; see the class file: [LDAGroupedGibbsSampler.java](src/main/java/cc/mallet/topics/LDAGroupedGibbsSampler.java)

4. **Approximate Distributed LDA** (ADLDA, [Newman et al. 2009](#5)). See the class file: [ADLDA.java](src/main/java/cc/mallet/topics/ADLDA.java)

## Building, installation, and execution


Install Apache [Maven](https://maven.apache.org/). To build the source code and package, use the following command in `bash` (it skips tests) 

    mvn -T 1C package -DskipTests -Dmaven.test.skip=true

or 

    "/opt/apache-maven-3.6.3/bin/mvn" -T 1C package -DskipTests -Dmaven.test.skip=true,

when  `mvn` (Version 3.6.3) is unavailable in the path. Modify the path according to the `mvn` installation. Modify the `mvn` configuration file [pom.xml](pom.xml) if required. This command creates a jar file `PCPLDA-X.X.X.jar` in the `target` folder (as specified in the configuration file: [pom.xml](pom.xml)).

To execute, prepare a configuration file; see [Configuration-README.txt](src/main/resources/configuration/Configuration-README.txt) for more details. The [configuration](src/main/resources/configuration) folder has a few example configuration files. The `dataset` folder has different test corpora. A stopwords file (`stopwords.txt`) is available in the repository. 

To run, use, for example. 

    java -jar target/pLDA-8.1.0.jar --run_cfg="plda-cats-test.cfg"

Note: plda-cats-test.cfg is available in [configuration](src/main/resources/configuration). 



## Acknowledgements

We thank the authors of [PartiallyCollapsedLDA](https://github.com/lejon/PartiallyCollapsedLDA) for making the source code and corpora publically available. 

## References 

<a id="1">Magnusson, M., Jonsson, L., Villani, M., & Broman, D. (2018)</a>. Sparse partially collapsed MCMC for parallel inference in topic models. Journal of Computational and Graphical Statistics, 27(2), 449-463.
  
<a id="2">Terenin, A., Magnusson, M., Jonsson, L., & Draper, D. (2018)</a>. Polya Urn Latent Dirichlet Allocation: a doubly sparse massively parallel sampler. IEEE transactions on pattern analysis and machine intelligence, 41(7), 1709-1719.

<a id="3">Doss, H. and George, C. (2022)</a>. Theoretical and Empirical Evaluation of a Grouped Gibbs Sampler for Parallel Computation in the Latent Dirichlet Allocation Model. In preparation. 

<a id="4">Griffiths, T. L., & Steyvers, M. (2004)</a>. Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl_1), 5228-5235.

<a id="5">Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009)</a>. Distributed algorithms for topic models. Journal of Machine Learning Research, 10(8).

<a id="6">Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003)</a>. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.