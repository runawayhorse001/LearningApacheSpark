.. _manipulation:

===========================
Data Manipulation: Features
===========================

.. admonition:: Chinese proverb

   **All things are diffcult before they are easy!** 

Feature building is a super important step for modeling which will determine the success or failure of your model. Otherwise, you will get: garbage in; garbage out! The techniques have been covered in the following chapters, the followings are the brief summary. I recently found that the Spark official website did a really good job for tutorial documentation. The chapter is based on  `Extracting transforming and selecting features`_. 

Feature Extraction
++++++++++++++++++

TF-IDF
------

Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. More details can be found at: https://spark.apache.org/docs/latest/ml-features#feature-extractors

`Stackoverflow TF`_: Both HashingTF and CountVectorizer can be used to generate the term frequency vectors. A few important differences:

a. partially **reversible** (CountVectorizer) vs **irreversible** (HashingTF) - since hashing is not reversible you cannot restore original input from a hash vector. From the other hand count vector with model (index) can be used to restore unordered input. As a consequence models created using hashed input can be much harder to interpret and monitor.
b. **memory and computational overhead** - HashingTF requires only a single data scan and no additional memory beyond original input and vector. CountVectorizer requires additional scan over the data to build a model and additional memory to store vocabulary (index). In case of unigram language model it is usually not a problem but in case of higher n-grams it can be prohibitively expensive or not feasible.
c. hashing **depends on** a size of the vector , hashing function and a document. Counting depends on a size of the vector, training corpus and a document.
d. **a source of the information loss** - in case of HashingTF it is dimensionality reduction with possible collisions. CountVectorizer discards infrequent tokens. How it affects downstream models depends on a particular use case and data.

1. HashingTF

`Stackoverflow TF`_: HashingTF is a Transformer which takes sets of terms and converts those sets into fixed-length feature vectors. In text processing, a “set of terms” might be a bag of words. HashingTF utilizes the hashing trick. A raw feature is mapped into an index (term) by applying a hash function. The hash function used here is MurmurHash 3. Then term frequencies are calculated based on the mapped indices. This approach avoids the need to compute a global term-to-index map, which can be expensive for a large corpus, but it suffers from potential hash collisions, where different raw features may become the same term after hashing. 

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import HashingTF, IDF, Tokenizer

	sentenceData = spark.createDataFrame([
	    (0.0, "I love Spark"),
	    (0.0, "I love python"),
	    (1.0, "I think ML is awesome")],
	 ["label", "sentence"])

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	vectorizer  = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)

	idf = IDF(inputCol="rawFeatures", outputCol="features")

	pipeline = Pipeline(stages=[tokenizer, vectorizer])


	model = pipeline.fit(sentenceData)
	model.transform(sentenceData).show(truncate=False)

.. code-block:: python

	+-----+---------------------+---------------------------+--------------------------------------------+
	|label|sentence             |words                      |rawFeatures                                 |
	+-----+---------------------+---------------------------+--------------------------------------------+
	|0.0  |I love Spark         |[i, love, spark]           |(100,[5,29,40],[1.0,1.0,1.0])               |
	|0.0  |I love python        |[i, love, python]          |(100,[29,40,89],[1.0,1.0,1.0])              |
	|1.0  |I think ML is awesome|[i, think, ml, is, awesome]|(100,[24,29,59,64,81],[1.0,1.0,1.0,1.0,1.0])|
	+-----+---------------------+---------------------------+--------------------------------------------+

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import HashingTF, IDF, Tokenizer

	sentenceData = spark.createDataFrame([
	    (0.0, "I love Spark"),
	    (0.0, "I love python"),
	    (1.0, "I think ML is awesome")],
	 ["label", "sentence"])

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	vectorizer  = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)

	idf = IDF(inputCol="rawFeatures", outputCol="features")

	pipeline = Pipeline(stages=[tokenizer, vectorizer,idf])


	model = pipeline.fit(sentenceData)
	model.transform(sentenceData).show(truncate=False)

.. code-block:: python

	+-----+---------------------+---------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------+
	|label|sentence             |words                      |rawFeatures                                 |features                                                                                                |
	+-----+---------------------+---------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------+
	|0.0  |I love Spark         |[i, love, spark]           |(100,[5,29,40],[1.0,1.0,1.0])               |(100,[5,29,40],[0.6931471805599453,0.0,0.28768207245178085])                                            |
	|0.0  |I love python        |[i, love, python]          |(100,[29,40,89],[1.0,1.0,1.0])              |(100,[29,40,89],[0.0,0.28768207245178085,0.6931471805599453])                                           |
	|1.0  |I think ML is awesome|[i, think, ml, is, awesome]|(100,[24,29,59,64,81],[1.0,1.0,1.0,1.0,1.0])|(100,[24,29,59,64,81],[0.6931471805599453,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
	+-----+---------------------+---------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------+

.. code-block:: python

	model.transform(sentenceData).select('features').show(truncate=False)

	+--------------------------------------------------------------------------------------------------------+
	|features                                                                                                |
	+--------------------------------------------------------------------------------------------------------+
	|(100,[5,29,40],[0.6931471805599453,0.0,0.28768207245178085])                                            |
	|(100,[29,40,89],[0.0,0.28768207245178085,0.6931471805599453])                                           |
	|(100,[24,29,59,64,81],[0.6931471805599453,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
	+--------------------------------------------------------------------------------------------------------+


2. Countvectorizer

`Stackoverflow TF`_: CountVectorizer and CountVectorizerModel aim to help convert a collection of text documents to vectors of token counts. When an a-priori dictionary is not available, CountVectorizer can be used as an Estimator to extract the vocabulary, and generates a CountVectorizerModel. The model produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms like LDA.

Countvectorizer
---------------

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import CountVectorizer
	from pyspark.ml.feature import HashingTF, IDF, Tokenizer

	sentenceData = spark.createDataFrame([
	    (0.0, "I love Spark"),
	    (0.0, "I love python"),
	    (1.0, "I think ML is awesome")],
	 ["label", "sentence"])

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	vectorizer  = CountVectorizer(inputCol="words", outputCol="rawFeatures")

	idf = IDF(inputCol="rawFeatures", outputCol="features")

	pipeline = Pipeline(stages=[tokenizer, vectorizer])


	model = pipeline.fit(sentenceData)
	model.transform(sentenceData).show(truncate=False)

.. code-block:: python

	+-----+---------------------+---------------------------+-------------------------------------+
	|label|sentence             |words                      |rawFeatures                          |
	+-----+---------------------+---------------------------+-------------------------------------+
	|0.0  |I love Spark         |[i, love, spark]           |(8,[0,1,2],[1.0,1.0,1.0])            |
	|0.0  |I love python        |[i, love, python]          |(8,[0,1,4],[1.0,1.0,1.0])            |
	|1.0  |I think ML is awesome|[i, think, ml, is, awesome]|(8,[0,3,5,6,7],[1.0,1.0,1.0,1.0,1.0])|
	+-----+---------------------+---------------------------+-------------------------------------+


.. code-block:: python

	counts = model.transform(sentenceData).select('rawFeatures').collect()
	counts

	[Row(rawFeatures=SparseVector(8, {0: 1.0, 1: 1.0, 2: 1.0})),
	 Row(rawFeatures=SparseVector(8, {0: 1.0, 1: 1.0, 4: 1.0})),
	 Row(rawFeatures=SparseVector(8, {0: 1.0, 3: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}))]

.. code-block:: python

	import numpy as np

	total_counts = model.transform(sentenceData)\
	                    .select('rawFeatures').rdd\
	                    .map(lambda row: row['rawFeatures'].toArray())\
	                    .reduce(lambda x,y: [x[i]+y[i] for i in range(len(y))])

	vocabList = model.stages[1].vocabulary
	d = {'vocabList':vocabList,'counts':total_counts}

	spark.createDataFrame(np.array(list(d.values())).T.tolist(),list(d.keys())).show()

.. code-block:: python

	+---------+------+
	|vocabList|counts|
	+---------+------+
	|        i|   3.0|
	|     love|   2.0|
	|       is|   1.0|
	|    think|   1.0|
	|       ml|   1.0|
	|  awesome|   1.0|
	|   python|   1.0|
	|    spark|   1.0|
	+---------+------+

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import CountVectorizer
	from pyspark.ml.feature import HashingTF, IDF, Tokenizer

	sentenceData = spark.createDataFrame([
	    (0.0, "I love Spark"),
	    (0.0, "I love python"),
	    (1.0, "I think ML is awesome")],
	 ["label", "sentence"])

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	vectorizer  = CountVectorizer(inputCol="words", outputCol="rawFeatures")

	idf = IDF(inputCol="rawFeatures", outputCol="features")

	pipeline = Pipeline(stages=[tokenizer, vectorizer,idf])


	model = pipeline.fit(sentenceData)
	model.transform(sentenceData).show()


.. code-block:: python

	+-----+---------------------+---------------------------+-------------------------------------+-------------------------------------------------------------------------------------------------+
	|label|sentence             |words                      |rawFeatures                          |features                                                                                         |
	+-----+---------------------+---------------------------+-------------------------------------+-------------------------------------------------------------------------------------------------+
	|0.0  |I love Spark         |[i, love, spark]           |(8,[0,1,5],[1.0,1.0,1.0])            |(8,[0,1,5],[0.0,0.28768207245178085,0.6931471805599453])                                         |
	|0.0  |I love python        |[i, love, python]          |(8,[0,1,6],[1.0,1.0,1.0])            |(8,[0,1,6],[0.0,0.28768207245178085,0.6931471805599453])                                         |
	|1.0  |I think ML is awesome|[i, think, ml, is, awesome]|(8,[0,2,3,4,7],[1.0,1.0,1.0,1.0,1.0])|(8,[0,2,3,4,7],[0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
	+-----+---------------------+---------------------------+-------------------------------------+-------------------------------------------------------------------------------------------------+

.. code-block:: python

	+-------------------------------------------------------------------------------------------------+
	|features                                                                                         |
	+-------------------------------------------------------------------------------------------------+
	|(8,[0,1,3],[0.0,0.28768207245178085,0.6931471805599453])                                         |
	|(8,[0,1,5],[0.0,0.28768207245178085,0.6931471805599453])                                         |
	|(8,[0,2,4,6,7],[0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])|
	+-------------------------------------------------------------------------------------------------+



Word2Vec
--------


.. code-block:: python

	from pyspark.ml.feature import Word2Vec

	from pyspark.ml import Pipeline

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
	word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="feature")

	pipeline = Pipeline(stages=[tokenizer, word2Vec])


	model = pipeline.fit(sentenceData)
	result = model.transform(sentenceData)


.. code-block:: python

	result.show()
	+-----+--------------------+--------------------+--------------------+
	|label|            sentence|               words|             feature|
	+-----+--------------------+--------------------+--------------------+
	|  0.0|        I love Spark|    [i, love, spark]|[0.05594437588782...|
	|  0.0|       I love python|   [i, love, python]|[-0.0350368790871...|
	|  1.0|I think ML is awe...|[i, think, ml, is...|[0.01242086507845...|
	+-----+--------------------+--------------------+--------------------+

.. code-block:: python

	w2v = model.stages[1]
	w2v.getVectors().show()

+-------+-----------------------------------------------------------------+
|word   |vector                                                           |
+-------+-----------------------------------------------------------------+
|is     |[0.13657838106155396,0.060924094170331955,-0.03379475697875023]  |
|awesome|[0.037024181336164474,-0.023855900391936302,0.0760037824511528]  |
|i      |[-0.0014482572441920638,0.049365971237421036,0.12016955763101578]|
|ml     |[-0.14006119966506958,0.01626444421708584,0.042281970381736755]  |
|spark  |[0.1589149385690689,-0.10970081388950348,-0.10547549277544022]   |
|think  |[0.030011219903826714,-0.08994936943054199,0.16471518576145172]  |
|love   |[0.01036644633859396,-0.017782460898160934,0.08870164304971695]  |
|python |[-0.11402882635593414,0.045119188725948334,-0.029877422377467155]|
+-------+-----------------------------------------------------------------+

.. code-block:: python

	from pyspark.sql.functions import format_number as fmt
	w2v.findSynonyms("could", 2).select("word", fmt("similarity", 5).alias("similarity")).show()

.. code-block:: python

	+-------+----------+
	|   word|similarity|
	+-------+----------+
	|classes|   0.90232|
	|      i|   0.75424|
	+-------+----------+


FeatureHasher
-------------

.. code-block:: python

	from pyspark.ml.feature import FeatureHasher

	dataset = spark.createDataFrame([
	    (2.2, True, "1", "foo"),
	    (3.3, False, "2", "bar"),
	    (4.4, False, "3", "baz"),
	    (5.5, False, "4", "foo")
	], ["real", "bool", "stringNum", "string"])

	hasher = FeatureHasher(inputCols=["real", "bool", "stringNum", "string"],
	                       outputCol="features")

	featurized = hasher.transform(dataset)
	featurized.show(truncate=False)

.. code-block:: python

	+----+-----+---------+------+--------------------------------------------------------+
	|real|bool |stringNum|string|features                                                |
	+----+-----+---------+------+--------------------------------------------------------+
	|2.2 |true |1        |foo   |(262144,[174475,247670,257907,262126],[2.2,1.0,1.0,1.0])|
	|3.3 |false|2        |bar   |(262144,[70644,89673,173866,174475],[1.0,1.0,1.0,3.3])  |
	|4.4 |false|3        |baz   |(262144,[22406,70644,174475,187923],[1.0,1.0,4.4,1.0])  |
	|5.5 |false|4        |foo   |(262144,[70644,101499,174475,257907],[1.0,1.0,5.5,1.0]) |
	+----+-----+---------+------+--------------------------------------------------------+


RFormula
--------

.. code-block:: python

	from pyspark.ml.feature import RFormula

	dataset = spark.createDataFrame(
	    [(7, "US", 18, 1.0),
	     (8, "CA", 12, 0.0),
	     (9, "CA", 15, 0.0)],
	    ["id", "country", "hour", "clicked"])

	formula = RFormula(
	    formula="clicked ~ country + hour",
	    featuresCol="features",
	    labelCol="label")

	output = formula.fit(dataset).transform(dataset)
	output.select("features", "label").show()

.. code-block:: python

	+----------+-----+
	|  features|label|
	+----------+-----+
	|[0.0,18.0]|  1.0|
	|[1.0,12.0]|  0.0|
	|[1.0,15.0]|  0.0|
	+----------+-----+



Feature Transform
+++++++++++++++++

Tokenizer
---------

.. code-block:: python

	from pyspark.ml.feature import Tokenizer, RegexTokenizer
	from pyspark.sql.functions import col, udf
	from pyspark.sql.types import IntegerType

	sentenceDataFrame = spark.createDataFrame([
	    (0, "Hi I heard about Spark"),
	    (1, "I wish Java could use case classes"),
	    (2, "Logistic,regression,models,are,neat")
	], ["id", "sentence"])

	tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

	regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
	# alternatively, pattern="\\w+", gaps(False)

	countTokens = udf(lambda words: len(words), IntegerType())

	tokenized = tokenizer.transform(sentenceDataFrame)
	tokenized.select("sentence", "words")\
	    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

	regexTokenized = regexTokenizer.transform(sentenceDataFrame)
	regexTokenized.select("sentence", "words") \
	    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

.. code-block:: python

	+-----------------------------------+------------------------------------------+------+
	|sentence                           |words                                     |tokens|
	+-----------------------------------+------------------------------------------+------+
	|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
	|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
	|Logistic,regression,models,are,neat|[logistic,regression,models,are,neat]     |1     |
	+-----------------------------------+------------------------------------------+------+

	+-----------------------------------+------------------------------------------+------+
	|sentence                           |words                                     |tokens|
	+-----------------------------------+------------------------------------------+------+
	|Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
	|I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
	|Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |5     |
	+-----------------------------------+------------------------------------------+------+

StopWordsRemover
----------------

.. code-block:: python

	from pyspark.ml.feature import StopWordsRemover

	sentenceData = spark.createDataFrame([
	    (0, ["I", "saw", "the", "red", "balloon"]),
	    (1, ["Mary", "had", "a", "little", "lamb"])
	], ["id", "raw"])

	remover = StopWordsRemover(inputCol="raw", outputCol="removeded")
	remover.transform(sentenceData).show(truncate=False)


.. code-block:: python

	+---+----------------------------+--------------------+
	|id |raw                         |removeded           |
	+---+----------------------------+--------------------+
	|0  |[I, saw, the, red, balloon] |[saw, red, balloon] |
	|1  |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
	+---+----------------------------+--------------------+


NGram
-----

.. code-block:: python

	from pyspark.ml.feature import NGram

	wordDataFrame = spark.createDataFrame([
	    (0, ["Hi", "I", "heard", "about", "Spark"]),
	    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
	    (2, ["Logistic", "regression", "models", "are", "neat"])
	], ["id", "words"])

	ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

	ngramDataFrame = ngram.transform(wordDataFrame)
	ngramDataFrame.select("ngrams").show(truncate=False)

.. code-block:: python

	+------------------------------------------------------------------+
	|ngrams                                                            |
	+------------------------------------------------------------------+
	|[Hi I, I heard, heard about, about Spark]                         |
	|[I wish, wish Java, Java could, could use, use case, case classes]|
	|[Logistic regression, regression models, models are, are neat]    |
	+------------------------------------------------------------------+

Binarizer
---------

.. code-block:: python

	from pyspark.ml.feature import Binarizer

	continuousDataFrame = spark.createDataFrame([
	    (0, 0.1),
	    (1, 0.8),
	    (2, 0.2),
	    (3,0.5)
	], ["id", "feature"])

	binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

	binarizedDataFrame = binarizer.transform(continuousDataFrame)

	print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
	binarizedDataFrame.show()


.. code-block:: python

	Binarizer output with Threshold = 0.500000
	+---+-------+-----------------+
	| id|feature|binarized_feature|
	+---+-------+-----------------+
	|  0|    0.1|              0.0|
	|  1|    0.8|              1.0|
	|  2|    0.2|              0.0|
	|  3|    0.5|              0.0|
	+---+-------+-----------------+

PCA
---

.. code-block:: python

	from pyspark.ml.feature import PCA
	from pyspark.ml.linalg import Vectors

	data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
	        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
	        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
	df = spark.createDataFrame(data, ["features"])

	pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
	model = pca.fit(df)

	result = model.transform(df).select("pcaFeatures")
	result.show(truncate=False)


.. code-block:: python

	+-----------------------------------------------------------+
	|pcaFeatures                                                |
	+-----------------------------------------------------------+
	|[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
	|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
	|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
	+-----------------------------------------------------------+

DCT
---

.. code-block:: python

	from pyspark.ml.feature import DCT
	from pyspark.ml.linalg import Vectors

	df = spark.createDataFrame([
	    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
	    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
	    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

	dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")

	dctDf = dct.transform(df)

	dctDf.select("featuresDCT").show(truncate=False)

.. code-block:: python

	+----------------------------------------------------------------+
	|featuresDCT                                                     |
	+----------------------------------------------------------------+
	|[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
	|[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
	|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
	+----------------------------------------------------------------+


StringIndexer
-------------


.. code-block:: python

	from pyspark.ml.feature import StringIndexer

	df = spark.createDataFrame(
	    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
	    ["id", "category"])

	indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
	indexed = indexer.fit(df).transform(df)
	indexed.show()


.. code-block:: python

	+---+--------+-------------+
	| id|category|categoryIndex|
	+---+--------+-------------+
	|  0|       a|          0.0|
	|  1|       b|          2.0|
	|  2|       c|          1.0|
	|  3|       a|          0.0|
	|  4|       a|          0.0|
	|  5|       c|          1.0|
	+---+--------+-------------+


labelConverter
--------------

.. code-block:: python

	from pyspark.ml.feature import IndexToString, StringIndexer

	df = spark.createDataFrame(
	    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
	    ["id", "label"])

	indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
	model = indexer.fit(df)
	indexed = model.transform(df)

	print("Transformed string column '%s' to indexed column '%s'"
	      % (indexer.getInputCol(), indexer.getOutputCol()))
	indexed.show()

	print("StringIndexer will store labels in output column metadata\n")

	converter = IndexToString(inputCol="labelIndex", outputCol="originalLabel")
	converted = converter.transform(indexed)

	print("Transformed indexed column '%s' back to original string column '%s' using "
	      "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
	converted.select("id", "labelIndex", "originalLabel").show()


.. code-block:: python

	Transformed string column 'label' to indexed column 'labelIndex'
	+---+-----+----------+
	| id|label|labelIndex|
	+---+-----+----------+
	|  0|    a|       0.0|
	|  1|    b|       2.0|
	|  2|    c|       1.0|
	|  3|    a|       0.0|
	|  4|    a|       0.0|
	|  5|    c|       1.0|
	+---+-----+----------+

	StringIndexer will store labels in output column metadata

	Transformed indexed column 'labelIndex' back to original string column 'originalLabel' using labels in metadata
	+---+----------+-------------+
	| id|labelIndex|originalLabel|
	+---+----------+-------------+
	|  0|       0.0|            a|
	|  1|       2.0|            b|
	|  2|       1.0|            c|
	|  3|       0.0|            a|
	|  4|       0.0|            a|
	|  5|       1.0|            c|
	+---+----------+-------------+


.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.feature import IndexToString, StringIndexer

	df = spark.createDataFrame(
	    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
	    ["id", "label"])

	indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
	converter = IndexToString(inputCol="labelIndex", outputCol="originalLabel")

	pipeline = Pipeline(stages=[indexer, converter])


	model = pipeline.fit(df)
	result = model.transform(df)

	result.show()

.. code-block:: python

	+---+-----+----------+-------------+
	| id|label|labelIndex|originalLabel|
	+---+-----+----------+-------------+
	|  0|    a|       0.0|            a|
	|  1|    b|       2.0|            b|
	|  2|    c|       1.0|            c|
	|  3|    a|       0.0|            a|
	|  4|    a|       0.0|            a|
	|  5|    c|       1.0|            c|
	+---+-----+----------+-------------+


VectorIndexer
-------------

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.regression import LinearRegression
	from pyspark.ml.feature import VectorIndexer
	from pyspark.ml.evaluation import RegressionEvaluator

	# Automatically identify categorical features, and index them.
	# We specify maxCategories so features with > 4 distinct values are treated as continuous.

	featureIndexer = VectorIndexer(inputCol="features", \
	                               outputCol="indexedFeatures",\
	                               maxCategories=4).fit(transformed)

	data = featureIndexer.transform(transformed)


.. code-block:: python

	data.show(5,True)

	+-----------------+-----+-----------------+
	|         features|label|  indexedFeatures|
	+-----------------+-----+-----------------+
	|[230.1,37.8,69.2]| 22.1|[230.1,37.8,69.2]|
	| [44.5,39.3,45.1]| 10.4| [44.5,39.3,45.1]|
	| [17.2,45.9,69.3]|  9.3| [17.2,45.9,69.3]|
	|[151.5,41.3,58.5]| 18.5|[151.5,41.3,58.5]|
	|[180.8,10.8,58.4]| 12.9|[180.8,10.8,58.4]|
	+-----------------+-----+-----------------+
	only showing top 5 rows

VectorAssembler
---------------

.. code-block:: python

	from pyspark.ml.linalg import Vectors
	from pyspark.ml.feature import VectorAssembler

	dataset = spark.createDataFrame(
	    [(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],
	    ["id", "hour", "mobile", "userFeatures", "clicked"])

	assembler = VectorAssembler(
	    inputCols=["hour", "mobile", "userFeatures"],
	    outputCol="features")

	output = assembler.transform(dataset)
	print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
	output.select("features", "clicked").show(truncate=False)

.. code-block:: python

	Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'
	+-----------------------+-------+
	|features               |clicked|
	+-----------------------+-------+
	|[18.0,1.0,0.0,10.0,0.5]|1.0    |
	+-----------------------+-------+

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


Normalizer
----------


.. code-block:: python

	from pyspark.ml.feature import Normalizer
	from pyspark.ml.linalg import Vectors

	dataFrame = spark.createDataFrame([
	    (0, Vectors.dense([1.0, 0.5, -1.0]),),
	    (1, Vectors.dense([2.0, 1.0, 1.0]),),
	    (2, Vectors.dense([4.0, 10.0, 2.0]),)
	], ["id", "features"])

	# Normalize each Vector using $L^1$ norm.
	normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
	l1NormData = normalizer.transform(dataFrame)
	print("Normalized using L^1 norm")
	l1NormData.show()

	# Normalize each Vector using $L^\infty$ norm.
	lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
	print("Normalized using L^inf norm")
	lInfNormData.show()

.. code-block:: python

	Normalized using L^1 norm
	+---+--------------+------------------+
	| id|      features|      normFeatures|
	+---+--------------+------------------+
	|  0|[1.0,0.5,-1.0]|    [0.4,0.2,-0.4]|
	|  1| [2.0,1.0,1.0]|   [0.5,0.25,0.25]|
	|  2|[4.0,10.0,2.0]|[0.25,0.625,0.125]|
	+---+--------------+------------------+

	Normalized using L^inf norm
	+---+--------------+--------------+
	| id|      features|  normFeatures|
	+---+--------------+--------------+
	|  0|[1.0,0.5,-1.0]|[1.0,0.5,-1.0]|
	|  1| [2.0,1.0,1.0]| [1.0,0.5,0.5]|
	|  2|[4.0,10.0,2.0]| [0.4,1.0,0.2]|
	+---+--------------+--------------+

Feature Selection
+++++++++++++++++

LASSO
-----

Variable selection and the removal of correlated variables.  The Ridge method shrinks the coefficients of correlated variables while the LASSO method picks one variable and discards the others.  The elastic net penalty is a mixture of these two; if variables are correlated in groups then :math:`\alpha=0.5` tends to select the groups as in or out. If α is close to 1, the elastic net performs much like the LASSO method and removes any degeneracies and wild behavior caused by extreme correlations. 



RandomForest
------------

`AutoFeatures`_ libray based on RandomForest is coming soon.............

.. _Extracting transforming and selecting features: https://spark.apache.org/docs/latest/ml-features
.. _Stackoverflow TF: https://stackoverflow.com/questions/35205865/what-is-the-difference-between-hashingtf-and-countvectorizer-in-spark
.. _OneHotEncoder: https://colab.research.google.com/drive/1pbrFQ-mcyijsVJNPP5GHbOeJaKdTLte3#scrollTo=kLU4xy3XLQG3
.. _AutoFeatures: https://github.com/runawayhorse001/AutoFeatures