.. _regularization:

==============
Regularization
==============


In mathematics, statistics, and computer science, particularly in the fields of machine learning and inverse problems,
regularization is a process of introducing additional information in order to solve an ill-posed problem or to prevent
overfitting (`Wikipedia Regularization`_).

Due to the sparsity within our data, our training sets will often be ill-posed (singular).  Applying regularization
to the regression has many advantages, including:

1. Converting ill-posed problems to well-posed by adding additional information via the
   penalty parameter :math:`\lambda`
2. Preventing overfitting
3. Variable selection and the removal of correlated variables (`Glmnet Vignette`_).  The Ridge method
   shrinks the coefficients of correlated variables while the LASSO method picks one variable and discards
   the others.  The elastic net penalty is a mixture of these two; if variables are correlated in groups then
   :math:`\alpha=0.5` tends to select the groups as in or out. If :math:`\alpha` is close to 1, the elastic
   net performs much like the LASSO method and removes any degeneracies and wild behavior caused by extreme
   correlations.


Ordinary least squares regression
+++++++++++++++++++++++++++++++++

.. math::

	\min _{\Bbeta\in \mathbb {R} ^{n}}{\frac {1}{n}}\|{\X}\Bbeta -{\y}\|^{2}

When :math:`\lambda=0` (i.e. ``regParam`` :math:`=0`), then there is no penalty.

.. code-block:: python

	LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, 
	regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, standardization=True, solver="auto", 
	weightCol=None, aggregationDepth=2)	


Ridge regression
++++++++++++++++

.. math::

	\min _{\Bbeta\in \mathbb {R} ^{n}}{\frac {1}{n}}\|{\X}\Bbeta-{\y}\|^{2}+\lambda \|\Bbeta\|_{2}^{2}

When :math:`\lambda>0` (i.e. ``regParam`` :math:`>0`) and :math:`\alpha=0` (i.e. ``elasticNetParam`` :math:`=0`)  , then the penalty is an L2 penalty.

.. code-block:: python

	LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, 
	regParam=0.1, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, standardization=True, solver="auto", 
	weightCol=None, aggregationDepth=2)	

Least Absolute Shrinkage and Selection Operator (LASSO)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. math::

	\min _{\Bbeta\in \mathbb {R} ^{n}}{\frac {1}{n}}\|{\X}\Bbeta-{\y}\|^{2}+\lambda\|\Bbeta\|_{1}

When :math:`\lambda>0` (i.e. ``regParam`` :math:`>0`) and :math:`\alpha=1` (i.e. ``elasticNetParam`` :math:`=1`), then the penalty is an L1 penalty.

.. code-block:: python

	LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, 
	regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, standardization=True, solver="auto", 
	weightCol=None, aggregationDepth=2)	

Elastic net
+++++++++++

.. math::

	\min _{\Bbeta\in \mathbb {R} ^{n}}{\frac {1}{n}}\|{\X}\Bbeta-{\y}\|^{2}+\lambda (\alpha \|\Bbeta\|_{1}+(1-\alpha )\|\Bbeta\|_{2}^{2}),\alpha \in (0,1)

When :math:`\lambda>0` (i.e. ``regParam`` :math:`>0`) and ``elasticNetParam`` :math:`\in (0,1)` (i.e. :math:`\alpha\in (0,1)`) , then the penalty is an L1 + L2 penalty.

.. code-block:: python

	LinearRegression(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, 
	regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, standardization=True, solver="auto", 
	weightCol=None, aggregationDepth=2)	


.. _Wikipedia Regularization: https://en.wikipedia.org/wiki/Regularization_(mathematics)
.. _Glmnet Vignette: https://web.stanford.edu/~hastie/Papers/Glmnet_Vignette.pdf


