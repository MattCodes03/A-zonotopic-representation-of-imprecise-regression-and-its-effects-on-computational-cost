# Zonotopic representation of multi-variable regression with interval dependent variables
*Matthew McCann and Marco De Angelis*,
*University of Strathclyde*

Reproducibility Repository for the Paper 'Zonotopic representation of multi-variable regression with interval dependent variables'. Contains all code used to produce the figures seen in the paper.

To run the examples please first ensure you have all the required libraries, as listed in `requirements.txt` to avoid any potential issues.

Paper - Link to come once published.

## Abstract
This paper expands on existing works on imprecise regression by shedding light on the set-valued representation of the computational problem of training regression models with interval-valued data. We frame the problem in a zonotopic sense to understand the computational cost and limitations of training imprecise regression models. 

We focus on the regression problem whereby the independent variable is measured without error while the dependent variable has uncertainty. Unlike classical regression, in imprecise regression the dependent variable is an interval-valued variable that is normally distributed.  We show that the problem can be reduced to a multi-dimensional Minkowski (dependent) sum of line segments, where each segment corresponds to an individual data interval. In other words, the problem is closed to zonotope representation.

This representation allows us to construct the feasible set of regression coefficients efficiently using affine transformations of interval boxes, rather than enumerating vertices as in traditional convex-hull approaches, avoiding exponential vertex enumeration by representing the zonotope compactly via generators. As a result, the zonotopic regression method provides a computationally efficient and interpretable middle ground between the overly conservative naive interval bounds and the exact but expensive vertex-based computation of tight bounds.

The advantages of this approach are that the uncertainty structure can be compactly stored through generator matrices and that it provides a clear geometric interpretation of parameter uncertainty, when working with interval-valued or imprecise representations of data. Since multilinear regression is closed under affine mappings, this formulation can be extended to nonlinear and neural network models, providing a framework for zonotopic uncertainty propagation. We demonstrate this approach on an emissions dataset, where precise values have been represented as intervals.
