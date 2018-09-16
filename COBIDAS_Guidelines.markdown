# Guidelines for the Reporting of Predictive Modeling Results

I would like to share the following ideas towards establishing a guideline/standard in reporting results on predictive modeling or model comparison, which I believe would improve the literature if followed.

## Context
The context for the suggested guideline is simple: when studies report some measure of predictive performance (such as accuracy, area under the curve, F1 score or similar), and/or try to claim their proposed method/biomarker outperforms another (or a set). Such claims so far, if supported, are mostly based on numerical differences (e.g. AUC=0.92 is better than AUC=0.90 - a case of model comparison when N=2), and sometimes without any significance testing. Even when significance testing is performed, the reviewers or readers don’t get to see the complete data/details on how it is performed. It gets murkier (for lack of a better word) when model comparison involves more than 2 models. Although machine learning literature provides methods/guidelines to handle this e.g. see [1], it appears that most neuroimagers are unfamiliar with such tests. Moreover we recently showed that common choices of cross-validation in neuroimaging can overestimate the performance, or with higher uncertainty [2].

## Proposal
In that context, I would like to propose the following:
 * that predictive performance (of any sort) be estimated with one of the community-accepted choices of cross-validation (e.g. repeated holdout with sufficiently large number N of random splits)
 * that performance be reported as a distribution (visually in a paper e.g. as a violinplot [3]), and as a shared data file (e.g. csv) providing the necessary data for other researchers to engage in proper model comparison (see below)
 * when claims of outperforming another method/feature are made, it must be substantiated by one of the community-accepted choices for significance testing. I propose non-parametric Friedman test with Nemenyi post-hoc analysis [1,3].
 * that we build a public repository (or web-based tool similar to neurovault or neurosynth) to store the published performances in a standardized format (that can be programmatically retrieved)
 * and to enable the authors (by providing all the tools/techniques necessary) via the aforementioned web-based tool to compare their new methods to an already existing method in appropriate ways.
 * For the novice users of machine learning, I am hoping to provide software that takes care of these non-trivial tasks and interface directly with the proposed repository, so they can focus on building better methods/biomarkers: see https://github.com/raamana/neuropredict

I might not have touched on similar use cases, and this is just one combination of suggestions. There might be other, or even better guidelines, that I might not be aware of, in which case I would be happy to learn and improve my proposal.

There are a number of choices we need to make, before proposing it, based on wide consultations with the community.

I would appreciate your feedback and support in helping me obtain wider feedback from the community.

## References
 1. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. The Journal of Machine Learning Research, 7, 1-30.
 2. Varoquaux, G., Raamana, P.R., Engemann, D.A., Hoyos-Idrobo, A., Schwartz, Y. and Thirion, B., 2017. Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. NeuroImage, 145, pp.166-179.
 3. Raamana, P.R. and Strother, S.C., 2016, June. Novel histogram-weighted cortical thickness networks and a multi-scale analysis of predictive power in Alzheimer's disease. In Pattern Recognition in Neuroimaging (PRNI), 2016 International Workshop on (pp. 1-4). IEEE.
