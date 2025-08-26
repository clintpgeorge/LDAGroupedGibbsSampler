# LDA Gibbs Samplers 

[Clint P. George](https://iitgoa.ac.in/~clint), [Hani Doss](https://users.stat.ufl.edu/~doss/) 

## Table of Contents
- [LDA Gibbs Samplers](#lda-gibbs-samplers)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Implemented Gibbs Samplers](#implemented-gibbs-samplers)
  - [Building, Installation, and Execution](#building-installation-and-execution)
    - [Prerequisites](#prerequisites)
    - [Build Instructions](#build-instructions)
    - [Execution Instructions](#execution-instructions)
  - [Acknowledgements](#acknowledgements)
  - [References](#references)

---

## Introduction
This project is forked from the repository [PartiallyCollapsedLDA](https://github.com/lejon/PartiallyCollapsedLDA). PartiallyCollapsedLDA implements parallel algorithms mentioned in 
[Magnusson et al. (2018)](#1) and [Terenin et al. (2018)](#2). For comparative study, a few other Gibbs samplers available in the literature, such as the Collapsed Gibbs Sampler ([CGS](#4), Griffiths and Steyvers, 2004), are implemented for the Latent Dirichlet Allocation ([LDA](#6)) model. 

The *main* Java `class` is `cc.mallet.topics.tui.ParallelLDA` ([ParallelLDA.java](src/main/java/cc/mallet/topics/tui/ParallelLDA.java)). This fork adds/edits the following Gibbs samplers, primarily to facilitate analyses in [Doss and George (2025)](#3): 

## Implemented Gibbs Samplers
1. **Collapsed Gibbs Sampler** (CGS, [Griffiths and Steyvers 2004](#4)): A serial implementation of CGS. See the class file: [SerialCollapsedLDA.java](src/main/java/cc/mallet/topics/SerialCollapsedLDA.java).
   
2. **Partially Collapsed Gibbs Sampler** (PCGS, [Magnusson et al. 2018](#1)): A parallel implementation of PCGS. See the class file: [UncollapsedParallelLDA.java](src/main/java/cc/mallet/topics/UncollapsedParallelLDA.java).

3. **Grouped Gibbs Sampler** (GGS, [Doss and George 2025](#3)): A parallel implementation of GGS. See the class file: [LDAGroupedGibbsSampler.java](src/main/java/cc/mallet/topics/LDAGroupedGibbsSampler.java).

4. **Approximate Distributed LDA** (ADLDA, [Newman et al. 2009](#5)): See the class file: [ADLDA.java](src/main/java/cc/mallet/topics/ADLDA.java).

---

## Building, Installation, and Execution

### Prerequisites
Install Apache [Maven](https://maven.apache.org/). Ensure `mvn` is available in your system's PATH or specify its full path during execution.

### Build Instructions
To build the source code and package it into a JAR file, use the following command in `bash` (skipping tests):

```bash
mvn -T 1C package -DskipTests -Dmaven.test.skip=true
```

If `mvn` is not in your PATH, use the full path to Maven, for example:

```bash
"/opt/apache-maven-3.6.3/bin/mvn" -T 1C package -DskipTests -Dmaven.test.skip=true
```

> **Note:** Modify the path according to your Maven installation. Update the Maven configuration file [pom.xml](pom.xml) if required.  
> The command creates a JAR file `PCPLDA-X.X.X.jar` in the `target` folder (as specified in [pom.xml](pom.xml)).

### Execution Instructions
1. Prepare a configuration file. See [Configuration-README.txt](src/main/resources/configuration/Configuration-README.txt) for details.  
   Example configuration files are available in the [configuration](src/main/resources/configuration) folder.  
   The `dataset` folder contains test corpora, and a stopwords file (`stopwords.txt`) is also included in the repository.

2. Run the program using the following command:

```bash
java -jar target/pLDA-8.1.0.jar --run_cfg="plda-cats-test.cfg"
```

> **Note:** The file `plda-cats-test.cfg` is available in the [configuration](src/main/resources/configuration) folder.

---

## Acknowledgements

We thank the authors of [PartiallyCollapsedLDA](https://github.com/lejon/PartiallyCollapsedLDA) for making the source code and corpora publicly available. 

---

## References 

<a id="1">Magnusson, M., Jonsson, L., Villani, M., & Broman, D. (2018)</a>. Sparse partially collapsed MCMC for parallel inference in topic models. *Journal of Computational and Graphical Statistics, 27*(2), 449–463. https://doi.org/10.1080/10618600.2017.1390473
  
<a id="2">Terenin, A., Magnusson, M., Jonsson, L., & Draper, D. (2018)</a>. Polya Urn Latent Dirichlet Allocation: A doubly sparse massively parallel sampler. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 41*(7), 1709–1719. https://doi.org/10.1109/TPAMI.2018.2841837

<a id="3">Doss, H., & George, C. (2025)</a>. Theoretical and empirical evaluation of a grouped Gibbs sampler for parallel computation in the Latent Dirichlet Allocation model. *In preparation.*

<a id="4">Griffiths, T. L., & Steyvers, M. (2004)</a>. Finding scientific topics. *Proceedings of the National Academy of Sciences, 101*(suppl_1), 5228–5235. https://doi.org/10.1073/pnas.0307752101

<a id="5">Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009)</a>. Distributed algorithms for topic models. *Journal of Machine Learning Research, 10*, 1801–1828. http://jmlr.org/papers/v10/newman09a.html

<a id="6">Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003)</a>. Latent Dirichlet Allocation. *Journal of Machine Learning Research, 3*(Jan), 993–1022. http://jmlr.org/papers/v3/blei03a.html