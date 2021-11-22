
.. _jdbc:

===============
JDBC Connection
===============

In this chapter, you will learn how to use JDBC source to write and read data in PySpark. The idea database for spark
is HDFS. But not many companies are not willing to move all the data (PII data) into Databricks' HDFS. Then you have
to use JDBC to connect to external database. While JDBC read and write are always tricky and confusion for beginner.


JDBC Driver
+++++++++++

For successfully connection, you need the corresponding JDBC driver for the specify Database. Here I will use Greenplum
database as an example to demonstrate how to get the correct ``.jar`` file and where to put the ``.jar``.

Get the ``.jar`` file
---------------------

Since Greenplum is using PostgreSQL, you can search with 'PostgreSQL JDBC Driver'. There is a high chance that
you will reach to this page: https://jdbc.postgresql.org/download.html. Then download the ``.jar`` file.

Put ``.jar`` in the jars folder
-------------------------------

Now what you need to do is putting the ``.jar`` file in the jar folder under your spark installation folder. Here is my
jar folder: ``/opt/spark/jars``

  .. _fig_jar:
  .. figure:: images/postgresql_driver.png
    :align: center

    JDBC connection jars folder


JDBC ``read``
+++++++++++++

See code `JDBC lower-upper Bound`_

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

JDBC ``write``
++++++++++++++

TODO...

JDBC ``temp_view``
+++++++++++++++++++
TODO...



.. _JDBC lower-upper Bound: https://github.com/apache/spark/blob/17edfec59de8d8680f7450b4d07c147c086c105a/sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/jdbc/JDBCRelation.scala#L85-L97
