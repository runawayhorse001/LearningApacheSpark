
.. _stats:

===========================================
Statistics and Linear Algebra Preliminaries
===========================================

.. admonition:: Chinese proverb

   **If you only know yourself, but not your opponent, you may win or may lose.
   If you know neither yourself nor your enemy, you will always endanger yourself** 
   – idiom, from Sunzi’s Art of War  


Notations 
+++++++++

* m : the number of the samples 
* n : the number of the features
* :math:`y_i` : i-th label 
* :math:`\hat{y}_i` : i-th predicted label
* :math:`{\displaystyle {\bar {\y}}} = {\frac {1}{m}}\sum _{i=1}^{m}y_{i}` :  the mean of :math:`\y`.
* :math:`\y` : the label vector.
* :math:`\hat{\y}` : the predicted label vector.


Linear Algebra Preliminaries
++++++++++++++++++++++++++++

Since I have documented the Linear Algebra Preliminaries in my Prelim Exam note for Numerical Analysis, the interested reader is referred to [Feng2014]_ for more details (Figure. :ref:`fig_linear_algebra`).

.. _fig_linear_algebra:
.. figure:: images/linear_algebra.png
   :align: center

   Linear Algebra Preliminaries


Measurement Formula
+++++++++++++++++++

Mean absolute error
-------------------

In statistics, **MAE** (`Mean absolute error`_) is a measure of difference between two continuous variables. The Mean Absolute Error is given by:

.. math::

	{\displaystyle \mathrm {MAE} ={\frac{1}{m} {\sum _{i=1}^{m}\left|\hat{y}_i-y_i\right|}}.}

Mean squared error
------------------

In statistics, the **MSE** (`Mean Squared Error`_) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors or deviations—that is, the difference between the estimator and what is estimated. 

.. math::

   \text{MSE}=\frac{1}{m}\sum_{i=1}^m\left( \hat{y}_i-y_i\right)^2  

Root Mean squared error
-----------------------

.. math::

   \text{RMSE} = \sqrt{\text{MSE}}=\sqrt{\frac{1}{m}\sum_{i=1}^m\left( \hat{y}_i-y_i\right)^2}    


Total sum of squares
--------------------

In statistical data analysis the **TSS** (`Total Sum of Squares`_) is a quantity that appears as part of a standard way of presenting results of such analyses. It is defined as being the sum, over all observations, of the squared differences of each observation from the overall mean.

.. math::

   \text{TSS} =  \sum_{i=1}^m\left( y_i-\bar{\y}\right)^2

Explained Sum of Squares
------------------------

In statistics, the **ESS** (`Explained sum of squares`_), alternatively known as the model sum of squares or sum of squares due to regression.

The ESS is the sum of the squares of the differences of the predicted values and the mean value of the response variable which is given by:

.. math::

   \text{ESS}= \sum_{i=1}^m\left( \hat{y}_i-\bar{\y}\right)^2 


Residual Sum of Squares
-----------------------

In statistics, **RSS** (`Residual sum of squares`_), also known as the sum of squared residuals (SSR) or the sum of squared errors of prediction (SSE), is the sum of the squares of residuals which is given by:

.. math::

   \text{RSS}= \sum_{i=1}^m\left( \hat{y}_i-y_i\right)^2 


Coefficient of determination :math:`R^2`
----------------------------------------

.. math::

	R^{2} := \frac{ESS}{TSS} = 1-{\text{RSS} \over \text{TSS}}.\,


.. note::

	In general (:math:`\y^{T}{\bar {\y}}={\hat {\y}}^{T}{\bar {\y}}`), total sum of squares = explained sum of squares + residual sum of squares, i.e.: 

	.. math::

		\text{TSS} = \text{ESS} + \text{RSS} \text{ if and only if } {\displaystyle \y^{T}{\bar {\y}}={\hat {\y}}^{T}{\bar {\y}}}.

	More details can be found at `Partitioning in the general ordinary least squares model`_. 	

Confusion Matrix
++++++++++++++++

.. _fig_con:
.. figure:: images/confusion_matrix.png
   :align: center

   Confusion Matrix

Recall
------

.. math::

   \text{Recall}=\frac{\text{TP}}{\text{TP+FN}}


Precision
---------

.. math::

   \text{Precision}=\frac{\text{TP}}{\text{TP+FP}}

Accuracy
--------

.. math::

   \text{Accuracy }=\frac{\text{TP+TN}}{\text{Total}}

:math:`F_1`-score
-----------------

.. math::

   \text{F}_1=\frac{2*\text{Recall}*\text{Precision}}{\text{Recall}+ \text{Precision}}


Statistical Tests
+++++++++++++++++

Correlational Test
------------------

* Pearson correlation: Tests for the strength of the association between two continuous variables.

* Spearman correlation: Tests for the strength of the association between two ordinal variables (does not rely on the assumption of normal distributed data).

* Chi-square: Tests for the strength of the association between two categorical variables.

Comparison of Means test
------------------------

* Paired T-test: Tests for difference between two related variables.

* Independent T-test: Tests for difference between two independent variables.

* ANOVA: Tests the difference between group means after any other variance in the outcome variable is accounted for.


Non-parametric Test
-------------------

* Wilcoxon rank-sum test: Tests for difference between two independent variables - takes into account magnitude and direction of difference.

* Wilcoxon sign-rank test: Tests for difference between two related variables - takes into account magnitude and direction of difference.

* Sign test: Tests if two related variables are different – ignores magnitude of change, only takes into account direction.

.. _Explained sum of squares: https://en.wikipedia.org/wiki/Explained_sum_of_squares
.. _Mean absolute error: https://en.wikipedia.org/wiki/Mean_absolute_error
.. _Residual sum of squares: https://en.wikipedia.org/wiki/Residual_sum_of_squares
.. _Mean Squared Error: https://en.wikipedia.org/wiki/Mean_squared_error
.. _Total Sum of Squares: https://en.wikipedia.org/wiki/Total_sum_of_squares
.. _Partitioning in the general ordinary least squares model: https://en.wikipedia.org/wiki/Explained_sum_of_squares