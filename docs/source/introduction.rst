Introduction
============
Seldonian RL library is an extension of the `Seldonian Framework <https://aisafety.cs.umass.edu>`_.
This work has been supervised by Professor Philip Thomas.

ML systems are influencing our lives more than ever before, but these systems can be imperfect, exhibiting racist and sexist
behavior, or recommending unsafe and even potentially fatal medical treatments.

This repository contains the implementation of Seldonian RL algorithms, which ensure high probability guarantees of safety and fairness.
Other implemented components are confidence intervals, off-policy policy evaluation methods, doubly robust estimator, some black box optimization algorithms, maximum likelihood modeling of the environment using the data, function approximation with neural network using lagrange multiplier (for constrained optimization).

The assumption for this problem is that the data is collected using some existing policy, and a new evaluation policy is learned from scratch using off-policy evaluation methods, without requiring the new policy to actually be used, since doing so could be dangerous or costly.