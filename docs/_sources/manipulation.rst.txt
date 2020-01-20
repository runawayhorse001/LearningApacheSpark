.. _manipulation:

===========================
Data Manipulation: Features
===========================

.. admonition:: Chinese proverb

   **All things are diffcult before they are easy!** 

Feature building is a super important step for modeling which will determine the success or failure of your model. Otherwise, you will get: garbage in; garbage out! The techniques have been covered in the following chapters, the followings are the brief summary.

Feature Extraction
++++++++++++++++++

Countvectorizer
---------------


TF-IDF
------

Word2Vec
--------

Feature Transform
+++++++++++++++++

StringIndexer
-------------

labelConverter
--------------


VectorIndexer
-------------

VectorAssembler
---------------


OneHotEncoder 
-------------

This is the note I wrote for one of my readers for explaining the OneHotEncoder. I would like to share it at here:

1. Import and creating SparkSession



.. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark create RDD example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()


.. code-block:: python

	df = spark.createDataFrame([
	    (0, "a"),
	    (1, "b"),
	    (2, "c"),
	    (3, "a"),
	    (4, "a"),
	    (5, "c")
	], ["id", "category"])
	df.show()

.. code-block:: python

	+---+--------+
	| id|category|
	+---+--------+
	|  0|       a|
	|  1|       b|
	|  2|       c|
	|  3|       a|
	|  4|       a|
	|  5|       c|
	+---+--------+


2. OneHotEncoder


a.  Encoder


.. code-block:: python

	from pyspark.ml.feature import OneHotEncoder, StringIndexer


	stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
	model = stringIndexer.fit(df)
	indexed = model.transform(df)

	# default setting: dropLast=True
	encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec",dropLast=False)
	encoded = encoder.transform(indexed)
	encoded.show()


.. code-block:: python


	+---+--------+-------------+-------------+
	| id|category|categoryIndex|  categoryVec|
	+---+--------+-------------+-------------+
	|  0|       a|          0.0|(3,[0],[1.0])|
	|  1|       b|          2.0|(3,[2],[1.0])|
	|  2|       c|          1.0|(3,[1],[1.0])|
	|  3|       a|          0.0|(3,[0],[1.0])|
	|  4|       a|          0.0|(3,[0],[1.0])|
	|  5|       c|          1.0|(3,[1],[1.0])|
	+---+--------+-------------+-------------+

.. note::

  The default setting of ``OneHotEncoder`` is: dropLast=True 

	.. code-block:: python

		# default setting: dropLast=True
		encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
		encoded = encoder.transform(indexed)
		encoded.show()


	.. code-block:: python

		+---+--------+-------------+-------------+
		| id|category|categoryIndex|  categoryVec|
		+---+--------+-------------+-------------+
		|  0|       a|          0.0|(2,[0],[1.0])|
		|  1|       b|          2.0|    (2,[],[])|
		|  2|       c|          1.0|(2,[1],[1.0])|
		|  3|       a|          0.0|(2,[0],[1.0])|
		|  4|       a|          0.0|(2,[0],[1.0])|
		|  5|       c|          1.0|(2,[1],[1.0])|
		+---+--------+-------------+-------------+

b. Vector Assembler

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import VectorAssembler
	categoricalCols = ['category']

	indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
	                 for c in categoricalCols ]
	# default setting: dropLast=True
	encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
	                 outputCol="{0}_encoded".format(indexer.getOutputCol()),dropLast=False)
	                 for indexer in indexers ]
	assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
	                            , outputCol="features")
	pipeline = Pipeline(stages=indexers + encoders + [assembler])

	model=pipeline.fit(df)
	data = model.transform(df)

.. code-block:: python

	data.show()
	+---+--------+----------------+------------------------+-------------+
	| id|category|category_indexed|category_indexed_encoded|     features|
	+---+--------+----------------+------------------------+-------------+
	|  0|       a|             0.0|           (3,[0],[1.0])|[1.0,0.0,0.0]|
	|  1|       b|             2.0|           (3,[2],[1.0])|[0.0,0.0,1.0]|
	|  2|       c|             1.0|           (3,[1],[1.0])|[0.0,1.0,0.0]|
	|  3|       a|             0.0|           (3,[0],[1.0])|[1.0,0.0,0.0]|
	|  4|       a|             0.0|           (3,[0],[1.0])|[1.0,0.0,0.0]|
	|  5|       c|             1.0|           (3,[1],[1.0])|[0.0,1.0,0.0]|
	+---+--------+----------------+------------------------+-------------+

3. Application: Get Dummy Variable

.. code-block:: python

	def get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol,dropLast=False):

	    '''
	    Get dummy variables and concat with continuous variables for ml modeling.
	    :param df: the dataframe
	    :param categoricalCols: the name list of the categorical data
	    :param continuousCols:  the name list of the numerical data
	    :param labelCol:  the name of label column
	    :param dropLast:  the flag of drop last column         
	    :return: feature matrix

	    :author: Wenqiang Feng
	    :email:  von198@gmail.com

	    >>> df = spark.createDataFrame([
	                  (0, "a"),
	                  (1, "b"),
	                  (2, "c"),
	                  (3, "a"),
	                  (4, "a"),
	                  (5, "c")
	              ], ["id", "category"])

	    >>> indexCol = 'id'
	    >>> categoricalCols = ['category']
	    >>> continuousCols = []
	    >>> labelCol = []

	    >>> mat = get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol)
	    >>> mat.show()

	    >>>
	        +---+-------------+
	        | id|     features|
	        +---+-------------+
	        |  0|[1.0,0.0,0.0]|
	        |  1|[0.0,0.0,1.0]|
	        |  2|[0.0,1.0,0.0]|
	        |  3|[1.0,0.0,0.0]|
	        |  4|[1.0,0.0,0.0]|
	        |  5|[0.0,1.0,0.0]|
	        +---+-------------+
	    '''

	    from pyspark.ml import Pipeline
	    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
	    from pyspark.sql.functions import col

	    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
	                 for c in categoricalCols ]

	    # default setting: dropLast=True
	    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
	                 outputCol="{0}_encoded".format(indexer.getOutputCol()),dropLast=dropLast)
	                 for indexer in indexers ]

	    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
	                                + continuousCols, outputCol="features")

	    pipeline = Pipeline(stages=indexers + encoders + [assembler])

	    model=pipeline.fit(df)
	    data = model.transform(df)

	    if indexCol and labelCol:
	        # for supervised learning
	        data = data.withColumn('label',col(labelCol))
	        return data.select(indexCol,'features','label')
	    elif not indexCol and labelCol:
	        # for supervised learning
	        data = data.withColumn('label',col(labelCol))
	        return data.select('features','label') 
	    elif indexCol and not labelCol:
	        # for unsupervised learning
	        return data.select(indexCol,'features')
	    elif not indexCol and not labelCol:
	        # for unsupervised learning
	        return data.select('features')      


a. Unsupervised scenario

.. code-block:: python

	df = spark.createDataFrame([
	    (0, "a"),
	    (1, "b"),
	    (2, "c"),
	    (3, "a"),
	    (4, "a"),
	    (5, "c")
	], ["id", "category"])
	df.show()

	indexCol = 'id'
	categoricalCols = ['category']
	continuousCols = []
	labelCol = []

	mat = get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol)


.. code-block:: python

	mat.show()

	+---+-------------+
	| id|     features|
	+---+-------------+
	|  0|[1.0,0.0,0.0]|
	|  1|[0.0,0.0,1.0]|
	|  2|[0.0,1.0,0.0]|
	|  3|[1.0,0.0,0.0]|
	|  4|[1.0,0.0,0.0]|
	|  5|[0.0,1.0,0.0]|
	+---+-------------+


b. Supervised scenario

.. code-block:: python

	df = spark.read.csv(path='bank.csv',
	                    sep=',',encoding='UTF-8',comment=None,
	                    header=True,inferSchema=True)

	indexCol = []
	catCols = ['job','marital','education','default',
	           'housing','loan','contact','poutcome']

	contCols = ['balance', 'duration','campaign','pdays','previous']
	labelCol = 'y'

	data = get_dummy(df,indexCol,catCols,contCols,labelCol,dropLast=False)
	data.show(5)

.. code-block:: python

	+--------------------+-----+
	|            features|label|
	+--------------------+-----+
	|(37,[8,12,17,19,2...|   no|
	|(37,[4,12,15,19,2...|   no|
	|(37,[0,13,16,19,2...|   no|
	|(37,[0,12,16,19,2...|   no|
	|(37,[1,12,15,19,2...|   no|
	+--------------------+-----+
	only showing top 5 rows


The Jupyter Notebook can be found on Colab: `OneHotEncoder`_ . 








Feature Selection
+++++++++++++++++


.. _OneHotEncoder: https://colab.research.google.com/drive/1pbrFQ-mcyijsVJNPP5GHbOeJaKdTLte3#scrollTo=kLU4xy3XLQG3