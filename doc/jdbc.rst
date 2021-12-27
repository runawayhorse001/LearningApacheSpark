
.. _jdbc:

===============
JDBC Connection
===============

In this chapter, you will learn how to use JDBC source to write and read data
in PySpark. The idea database for spark is HDFS. But not many companies are
not willing to move all the data (PII data) into Databricks' HDFS. Then you
have to use JDBC to connect to external database. While JDBC read and write
are always tricky and confusion for beginner.


JDBC Driver
+++++++++++

For successfully connection, you need the corresponding JDBC driver for the
specify Database. Here I will use Greenplum database as an example to
demonstrate how to get the correct ``.jar`` file and where to put the ``.jar``.

Get the ``.jar`` file
---------------------

Since Greenplum is using PostgreSQL, you can search with
'PostgreSQL JDBC Driver'. There is a high chance that
you will reach to this page: https://jdbc.postgresql.org/download.html.
Then download the ``.jar`` file.

Put ``.jar`` in the jars folder
-------------------------------

Now what you need to do is putting the ``.jar`` file in the jar folder under
your spark installation folder. Here is my
jar folder: ``/opt/spark/jars``

  .. _fig_jar:
  .. figure:: images/postgresql_driver.png
    :align: center

    JDBC connection jars folder

Get Credentials
+++++++++++++++

The following is the demo how to get the credentials in Azure Databricks:

.. code-block:: python

    def azdb_credentials(databaseName="DEFAULT_DB"):
        """
        Create credentials by usung dbutils.secrets

        :param databaseName: Database need to build the connection,
        :return: credentials
        """

        # get dbutils module
        dbutils = IPython.get_ipython().user_ns["dbutils"]

        # Read Secret Keys from Key-Vault
        service_principal_id = dbutils.secrets\
                                      .get(scope="secret-scope-key-vault",
                                           key="DB-SPN-ID")

        service_principal_secret = dbutils.secrets\
                                          .get(scope="secret-scope-key-vault",
                                               key="DB-SPN-SECRET")

        tenent_id = dbutils.secrets.get(scope="ecret-scope-key-vault",
                                        key="TENENT-ID")

        # define the authority URL and your tenant ID
        ##In Azure
        ###authorityHostUrl = "https://login.microsoftonline.com"
        authority_url = ('https://login.microsoftonline.com' + '/' + tenent_id)
        context = adal.AuthenticationContext(authority_url, api_version=None)
        token = context.acquire_token_with_client_credentials(
                        "https://database.windows.net/",
                        service_principal_id,
                        service_principal_secret)

        credentials = {'access_token': token["accessToken"],
                       'url': "jdbc:sqlserver://a3pcdsapsq01.database.windows.net"
                              + ";" + "databaseName=" + databaseName + ";"}

        return credentials

JDBC ``read``
+++++++++++++

The most tricky part for JDBC ``read`` is ensuring:
    1. parallel reading
    2. almost evenly distributed partition size for each partition

First Let's look at the source code `JDBC lower-upper Bound`_:

.. code-block:: python

    stride = (upper_bound/partitions_number) - (lower_bound/partitions_number)
    partition_nr = 0
    while (partition_nr < partitions_number)
    generate WHERE clause:
      partition_column IS NULL OR partition_column < stride
      if:
        partition_nr == 0 AND partition_nr < partitions_number
    or generate WHERE clause:
      partition_column &gt;= stride AND partition_column &lt;  next_stride
      if:
        partition_nr < 0 AND partition_nr &lt; partitions_number
    or generate WHERE clause
      partition_column >= stride
      if:
        partition_nr > 0 AND partition_nr == partitions_number
    where next_stride is calculated after computing the left sideo
    of the WHERE clause by next_stride += stride



    (stride = (20/5) - (0/5) = 4
    SELECT * FROM my_table WHERE partition_column IS NULL OR partition_column < 4
    SELECT * FROM my_table WHERE partition_column >= 4 AND partition_column < 8
    SELECT * FROM my_table WHERE partition_column >= 8 AND partition_column < 12
    SELECT * FROM my_table WHERE partition_column >= 12 AND partition_column < 16
    SELECT * FROM my_table WHERE partition_column >= 16

As you see, the above queries generate 5 partitions of data, each containing the values
from: (0-3), (4-7), (8-11), (12-15) and (16 and more).

.. warning::

    The above results is only true when your partition column has incremental
    like column. If you are using ``DB2``, ``Netezza`` or ``Oracle``, they have
    implicit ``rowid`` you can use it as ``partitionColumn``. Otherwise, you'd
    better to work with your engineer team to add the ``rowid`` to each table
    (this is the most convenient way to fix the problem in this part).

.. code-block:: python

    def azdb_read(table_name, partition_column='rowid', lower_bound=0,
                  upper_bound=1000000, num_partitions=200, fetchsize=5000,
                  database_name="DEFAULT_DB"):
        """
        Read table from DEFAULT_DB.

        :param table_name: table to be read, ONLY need table name at here
        :param partition_column: column name used to do partition, must be a
                                 numeric, date, or timestamp
        :param lower_bound: lower bound of values to be fetched
        :param upper_bound: upper bound of values to be fetched
        :param num_partitions: The maximum number of partitions that can be used
                               for parallelism in table reading
        :param fetchsize: The JDBC fetch size, which determines how many rows to
                          fetch per round trip. This can help performance on JDBC
                          drivers which default to low fetch size (e.g. Oracle
                          with 10 rows).This option applies only to reading.
        :param database_name: The source database where the table will be in.
        :return:  DataFrame
        """

        spark = SparkSession.builder.getOrCreate()
        credentials = azdb_credentials(database_name)

        df = spark.read \
            .format("com.microsoft.sqlserver.jdbc.spark") \
            .option("url", credentials['url']) \
            .option("dbtable", table_name) \
            .option("accessToken", credentials['access_token']) \
            .option('partitionColumn', partition_column) \
            .option('numPartitions', num_partitions) \
            .option("lowerBound", lower_bound) \
            .option("upperBound", upper_bound) \
            .option("fetchsize", fetchsize) \
            .option("encrypt", "true") \
            .option("hostNameInCertificate", "*.database.windows.net") \
            .load()

        return df

JDBC ``write``
++++++++++++++

TODO...

.. code-block:: python

    def azdb_write(df, table_name, num_partitions=200, write_mode='error',
                   batchsize=5000, database_name="DEFAULT_DB"):
        """
        Write spark DataFrame back to DEFAULT_DB

        :param df: the spark DataFrame need to be written
        :param table_name: the name of the table in DEFAULT_DB, ONLY need table
                           name at here
        :param num_partitions: The maximum number of partitions that can be used
                               for parallelism in table writing
        :param write_mode: 'error'(default), 'append', 'overwrite', 'ignore'
        :param batchsize: The JDBC batch size, which determines how many rows to
                          insert per round trip. This can help performance on JDBC
                          drivers. This option applies only to writing.
                          It defaults to 1000.
        :param database_name: The designated database where the table will be in.
        :return: None
        """
        credentials = azdb_credentials(database_name)

        df.write \
          .mode(write_mode) \
          .format("com.microsoft.sqlserver.jdbc.spark") \
          .option("url", credentials['url']) \
          .option("dbtable", table_name) \
          .option("accessToken", credentials['access_token']) \
          .option('numPartitions', num_partitions) \
          .option("batchsize", batchsize) \
          .option("encrypt", "true") \
          .option("hostNameInCertificate", "*.database.windows.net") \
          .save()

.. warning::

    If you are working with some database which is not supporting ``string``
    datatype, such as ``DB2``, ``Netezza``, you need pass the custom schema as
    an option like

    .. code-block:: python

        .option("createTableColumnTypes", custom_schema)

    Actually, you can use the following code to automatically generate the
    custom schema:

    .. code-block:: python

        def auto_schema(df):
            cols_string = [k for k, v in dict(df.dtypes).items() if v=='string' ]
            features_len = df.select(*(F.length(F.col(c)).alias(c) for c in
                                       cols_string)).cache()

            max_lenth ={}
            for c in cols_string:
                max_lenth[c] = features_len.agg(F.max(F.col(c))).collect()[0][0]

            schema_dict = [(k, 'VARCHAR({})'.format(v)) for k,v in max_lenth.items()]

            return ', '.join(i for i in [' '.join([j for j in i]) for i in schema_dict])


JDBC ``temp_view``
+++++++++++++++++++
TODO...

.. code-block:: python

    def azdb_temp_view(table_name, partition_column='rowid', lower_bound=0,
                       upper_bound=1000000, num_partitions=200,
                       fetchsize=5000, database_name="DEFAULT_DB",
                       temp_table_name=None):
        """
        Read a table in DEFAULT_DB with pyspark and create global temporary view

        :param table_name: database table to be read
        :param temp_table_name: temporary table name
        :param partition_column: column name used to do partition, must be a
                                 numeric, date, or timestamp
        :param lower_bound: lower bound of values to be fetched
        :param upper_bound: upper bound of values to be fetched
        :param num_partitions: The maximum number of partitions that can be
                               used for parallelism in table reading
        :param database_name: default DEFAULT_DB
        :return: pyspark dataframe for the read table
        """

        spark = SparkSession.builder.getOrCreate()
        output = azdb_read(table_name, partition_column, lower_bound,
                           upper_bound, num_partitions, fetchsize, database_name)

        if temp_table_name:
            spark.catalog.dropGlobalTempView(temp_table_name.upper())
            output.createGlobalTempView(temp_table_name.upper())
        else:
            spark.catalog.dropGlobalTempView(table_name.upper().split('.')[-1])
            output.createGlobalTempView(table_name.upper().split('.')[-1])

        return output



.. _JDBC lower-upper Bound: https://github.com/apache/spark/blob/17edfec59de8d8680f7450b4d07c147c086c105a/sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/jdbc/JDBCRelation.scala#L85-L97
