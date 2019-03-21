
.. _why:

========================
Why Spark with Python ? 
========================

.. admonition:: Chinese proverb

  **Sharpening the knife longer can make it easier to hack the firewood** -- old Chinese proverb

I want to answer this question from the following two parts:

Why Spark?
++++++++++

I think the following four main reasons from `Apache Spark™`_ official website are good enough
to convince you to use Spark.

1. Speed

   Run programs up to 100x faster than Hadoop MapReduce in memory, or 10x faster on disk.

   Apache Spark has an advanced DAG execution engine that supports acyclic data flow and in-memory computing.

  .. _fig_lr:
  .. figure:: images/logistic-regression.png
    :align: center

    Logistic regression in Hadoop and Spark



2. Ease of Use

   Write applications quickly in Java, Scala, Python, R.

   Spark offers over 80 high-level operators that make it easy to build parallel apps. And you can use it interactively from the Scala, Python and R shells.


3. Generality

   Combine SQL, streaming, and complex analytics.

   Spark powers a stack of libraries including SQL and DataFrames, MLlib for machine learning, GraphX, and Spark Streaming. You can combine these libraries seamlessly in the same application.	

  .. _fig_stack:
  .. figure:: images/stack.png
    :align: center
    :scale: 70 %

    The Spark stack

4. Runs Everywhere
   
   Spark runs on Hadoop, Mesos, standalone, or in the cloud. It can access diverse data sources including HDFS, Cassandra, HBase, and S3.

  .. _fig_runs:
  .. figure:: images/spark-runs-everywhere.png
    :align: center
    :scale: 60 %

    The Spark platform



Why Spark with Python (PySpark)?
++++++++++++++++++++++++++++++++

No matter you like it or not, Python has been one of the most popular programming languages.


  .. _fig_languages:
  .. figure:: images/languages.jpg
    :align: center

    KDnuggets Analytics/Data Science 2017 Software Poll from `kdnuggets`_.


.. _Apache Spark™ : http://spark.apache.org/

.. _kdnuggets: http://www.kdnuggets.com/2017/05/poll-analytics-data-science-machine-learning-software-leaders.html




