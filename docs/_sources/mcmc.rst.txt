.. _mcmc:

========================
Markov Chain Monte Carlo
========================

.. |theta| replace:: :math:`\theta`

.. admonition:: Chinese proverb

	**A book is known in time of need.**

.. figure:: images/mcmc_py.png
   :align: center

Monte Carlo simulations are just a way of estimating a fixed parameter by repeatedly generating random numbers. More details can be found at `A Zero Math Introduction to Markov Chain Monte Carlo Methods`_. 

Markov Chain Monte Carlo (MCMC) methods are used to approximate the posterior distribution of a parameter of interest by random sampling in a probabilistic space. More details can be found at `A Zero Math Introduction to Markov Chain Monte Carlo Methods`_. 

The following theory and demo are from Dr. Rebecca C. Steorts's `Intro to Markov Chain Monte Carlo`_. More details can be found at Dr. Rebecca C. Steorts's STA 360/601: `Bayesian Methods and Modern Statistics`_ class at Duke.

.. _metroalg:

Metropolis algorithm
++++++++++++++++++++

The Metropolis algorithm takes three main steps:

1. Sample  :math:`\theta^* \sim J(\theta | \theta ^{(s)})`

2. Compute the acceptance ratio :math:`(r)`

	.. math::

	        r = \frac{p(\theta^*|y)}{p(\theta^{(s)}|y)} = \frac{p(y|\theta^*)p(\theta^*)}{p(y|\theta^{(s)})p(\theta^{(s)})} 

3. Let

	.. math::
	   :label: eq_step3

			\theta^{(s+1)} 
			=
			\left\{
            	\begin{array}{ll}
                	\theta^* &\text{ with prob min}{(r,1)} \\
                	\theta^{(s)} &\text{ otherwise }
            	\end{array}
         	\right.

.. note::

   Actually, the :eq:`eq_step3` in Step 3 can be replaced by sampling :math:`u \sim \text{Uniform}(0,1)` and setting :math:`\theta^{(s+1)}=\theta^*` if :math:`u<r` and setting :math:`\theta^{(s+1)}=\theta^{(s)}` otherwise.


A Toy Example of Metropolis
+++++++++++++++++++++++++++


The following example is going to test out the Metropolis algorithm for the conjugate Normal-Normal model with a known variance situation. 

Conjugate Normal-Normal model
-----------------------------

	.. math::

            \begin{array}{ll}
                X_1, \cdots, X_n & \theta \stackrel{iid}{\sim}\text{Normal}(\theta,\sigma^2)\\
                                  & \theta \sim\text{Normal}(\mu,\tau^2)
            \end{array}

Recall that the posterior of |theta| is :math:`\text{Normal}(\mu_n,\tau^2_n)`, where

	.. math::

            \mu_n = \bar{x}\frac{n/\sigma^2}{n/\sigma^2+1/\tau^2} + \mu\frac{1/\tau^2}{n/\sigma^2+1/\tau^2}

and 

	.. math::

            \tau_n^2 = \frac{1}{n/\sigma^2+1/\tau^2}

Example setup
-------------

The rest of the parameters are :math:`\sigma^2=1`, :math:`\tau^2=10`, :math:`\mu=5`, :math:`n=5` and

	.. math::

            y = [9.37, 10.18, 9.16, 11.60, 10.33]

For this setup, we get that :math:`\mu_n=10.02745` and :math:`\tau_n^2=0.1960784`.

Essential mathematical derivation 
---------------------------------

In the :ref:`metroalg`, we need to compute the acceptance ratio :math:`r`, i.e.

	.. math::

		   r  &=  \frac{p(\theta^*|x)}{p(\theta^{(s)}|x)} \\
		      &=  \frac{p(x|\theta^*)p(\theta^*)}{p(x|\theta^{(s)})p(\theta^{(s)})}\\
		      &=  \left(\frac{\prod_i\text{dnorm}(x_i,\theta^*,\sigma)}{\prod_i\text{dnorm}(x_i,\theta^{(s)},\sigma)}\right) 
		           \left(\frac{\text{dnorm}(\theta^*,\mu,\tau)}{\text{dnorm}(\theta^{(s)},\mu,\tau)}\right) 

In many cases, computing the ratio :math:`r` directly can be numerically unstable, however, this can be modified by taking :math:`log r`. i.e.

	.. math::

		   logr  &=  \sum_i \left(log[\text{dnorm}(x_i,\theta^*,\sigma)] - log[\text{dnorm}(x_i, \theta^{(s)}, \sigma)]\right)\\
		         &+  \sum_i \left(log[\text{dnorm}(\theta^*,\mu,\tau)] - log[\text{dnorm}(\theta^{(s)}, \mu,\tau)]\right)
		     
Then the criteria of the acceptance becomes: if :math:`log u< log r`, where :math:`u` is sample form the :math:`\text{Uniform}(0,1)`.  

Demos
+++++

Now, We generate :math:`S` iterations of the Metropolis algorithm starting at :math:`\theta^{(0)}=0` and using a normal proposal distribution, where

	.. math::

		   \theta^{(s+1)} \sim \text{Normal}(\theta^{(s)},2).

R results
---------

.. literalinclude:: code/mcmc.R
     :language: r

.. _fig_mcmc_r:
.. figure:: images/mcmc_r.png
   :align: center

   Histogram for the Metropolis algorithm with r

Figure. :ref:`fig_mcmc_r` shows a trace plot for this run as well as a histogram for
the Metropolis algorithm compared with a draw from the true normal density.


Python results
--------------

.. literalinclude:: code/mcmc.py
     :language: python

.. _fig_mcmc_py:
.. figure:: images/mcmc_py.png
   :align: center

   Histogram for the Metropolis algorithm with python

Figure. :ref:`fig_mcmc_py` shows a trace plot for this run as well as a histogram for
the Metropolis algorithm compared with a draw from the true normal density.



PySpark results
---------------

TODO...

.. _fig_mcmc_pyspark:
.. figure:: images/mcmc_py.png
   :align: center

   Histogram for the Metropolis algorithm with PySpark

Figure. :ref:`fig_mcmc_pyspark` shows a trace plot for this run as well as a histogram for
the Metropolis algorithm compared with a draw from the true normal density.



.. _A Zero Math Introduction to Markov Chain Monte Carlo Methods: https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50
.. _Intro to Markov Chain Monte Carlo: http://www2.stat.duke.edu/~rcs46/lecturesModernBayes/601-module6-markov/markov-chain-monte-carlo.pdf
.. _Bayesian Methods and Modern Statistics: http://www2.stat.duke.edu/~rcs46/bayes.html