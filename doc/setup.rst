.. _setup:


==========================
Configure Running Platform
==========================

.. admonition:: Chinese proverb

  **Good tools are prerequisite to the successful execution 
  of a job.** -- old Chinese proverb

 
A good programming platform can save you lots of troubles and time. 
Herein I will only present how to install my favorite programming 
platform and only show the easiest way which I know to set it up 
on Linux system. If you want to install on the other operator 
system, you can Google it. In this section, you may learn how to
set up Pyspark on the corresponding programming platform and package.


.. index:: Run on Databricks Community Cloud

Run on Databricks Community Cloud
+++++++++++++++++++++++++++++++++

If you don't have any experience with Linux or Unix operator 
system, I would love to recommend you to use Spark on Databricks 
Community Cloud. Since you do not need to setup the Spark and it's
totally **free** for Community Edition. Please follow the steps
listed below.

 1. Sign up a account at: https://community.cloud.databricks.com/login.html 

  .. _fig_login:
  .. figure:: images/login.png
    :align: center

 2. Sign in with your account, then you can creat your cluster(machine), table(dataset)
    and notebook(code).  

  .. _fig_workspace:
  .. figure:: images/workspace.png
    :align: center

 3. Create your cluster where your code will run

  .. _fig_cluster:
  .. figure:: images/cluster.png
    :align: center

 4. Import your dataset

  .. _fig_table:
  .. figure:: images/table.png
    :align: center

  .. _fig_dataset1:
  .. figure:: images/dataset1.png
    :align: center

 .. note::
   
    You need to save the path which appears at Uploaded to DBFS:  
    /FileStore/tables/05rmhuqv1489687378010/. Since we will use
    this path to load the dataset.

5. Create your notebook

  .. _fig_notebook:
  .. figure:: images/notebook.png
    :align: center

  .. _fig_codenotebook:
  .. figure:: images/codenotebook.png
    :align: center

After finishing the above 5 steps, you are ready to run your
Spark code on Databricks Community Cloud. I will run all the 
following demos on Databricks Community Cloud. Hopefully, when
you run the demo code, you will get the following results:

 .. code-block:: python

	+---+-----+-----+---------+-----+
	|_c0|   TV|Radio|Newspaper|Sales|
	+---+-----+-----+---------+-----+
	|  1|230.1| 37.8|     69.2| 22.1|
	|  2| 44.5| 39.3|     45.1| 10.4|
	|  3| 17.2| 45.9|     69.3|  9.3|
	|  4|151.5| 41.3|     58.5| 18.5|
	|  5|180.8| 10.8|     58.4| 12.9|
	+---+-----+-----+---------+-----+
	only showing top 5 rows

	root
	 |-- _c0: integer (nullable = true)
	 |-- TV: double (nullable = true)
	 |-- Radio: double (nullable = true)
	 |-- Newspaper: double (nullable = true)
	 |-- Sales: double (nullable = true) 



.. index:: Configure Spark on Mac and Ubuntu

.. _set-up-Ubuntu:  

Configure Spark on Mac and Ubuntu
+++++++++++++++++++++++++++++++++

Installing Prerequisites
------------------------
  
I will strongly recommend you to install `Anaconda`_, since it contains most 
of the prerequisites and support multiple Operator Systems.
  
1. **Install Python**

Go to Ubuntu Software Center and follow the following steps:

  a. Open Ubuntu Software Center 
  b. Search for python
  c. And click Install

Or Open your terminal and  using the following command:

.. code-block:: bash

  sudo apt-get install build-essential checkinstall
  sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev 
                   libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
  sudo apt-get install python
  sudo easy_install pip
  sudo pip install ipython

Install Java
------------

Java is used by many other softwares. So it is quite possible that you have already installed it. You can 
by using the following command in Command Prompt:

.. code-block:: bash
 
  java -version 

Otherwise, you can follow the steps in `How do I install Java for my Mac?`_ to install java on Mac and use the following command in Command Prompt to install on Ubuntu:

.. code-block:: bash
 
  sudo apt-add-repository ppa:webupd8team/java
  sudo apt-get update
  sudo apt-get install oracle-java8-installer



Install Java SE Runtime Environment
-----------------------------------  

I installed ORACLE `Java JDK`_.  

.. warning::

  **Installing Java and Java SE Runtime Environment steps are very important, since Spark is a domain-specific language written in Java.**


You can check if your Java is available and find it’s version by using the following 
command in Command Prompt:

.. code-block:: bash
 
  java -version 

If your Java is installed successfully, you will get the similar results as follows:
     
.. code-block:: bash
 
  java version "1.8.0_131"
  Java(TM) SE Runtime Environment (build 1.8.0_131-b11)
  Java HotSpot(TM) 64-Bit Server VM (build 25.131-b11, mixed mode) 
  
Install Apache Spark
-------------------- 

Actually, the Pre-build version doesn’t need installation. You can use it when you unpack it.
   
  a. Download: You can get the Pre-built Apache Spark™ from `Download Apache Spark™`_. 
  b. Unpack: Unpack the Apache Spark™ to the path where you want to install the Spark.
  c. Test: Test the Prerequisites: change the direction ``spark-#.#.#-bin-hadoop#.#/bin`` and run

  .. code-block:: bash
 
   ./pyspark

  .. code-block:: bash
 
   Python 2.7.13 |Anaconda 4.4.0 (x86_64)| (default, Dec 20 2016, 23:05:08)
   [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   Anaconda is brought to you by Continuum Analytics.
   Please check out: http://continuum.io/thanks and https://anaconda.org
   Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
   Setting default log level to "WARN".
   To adjust logging level use sc.setLogLevel(newLevel). For SparkR, 
   use setLogLevel(newLevel).
   17/08/30 13:30:12 WARN NativeCodeLoader: Unable to load native-hadoop 
   library for your platform... using builtin-java classes where applicable
   17/08/30 13:30:17 WARN ObjectStore: Failed to get database global_temp, 
   returning NoSuchObjectException 
   Welcome to
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version 2.1.1
          /_/

   Using Python version 2.7.13 (default, Dec 20 2016 23:05:08)
   SparkSession available as 'spark'.

Configure the Spark
------------------- 

  a. **Mac Operator System:** open your ``bash_profile`` in Terminal

  .. code-block:: bash
 
   vim ~/.bash_profile
  
  And add the following lines to your ``bash_profile`` (remember to change the path)

  .. code-block:: bash
 
   # add for spark
   export SPARK_HOME=your_spark_installation_path
   export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
   export PATH=$PATH:$SPARK_HOME/bin
   export PYSPARK_DRIVER_PYTHON="jupyter"
   export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

  At last, remember to source your ``bash_profile``
   
  .. code-block:: bash
 
   source ~/.bash_profile 

  b. **Ubuntu Operator Sysytem:** open your ``bashrc`` in Terminal

  .. code-block:: bash
 
   vim ~/.bashrc
  
  And add the following lines to your ``bashrc`` (remember to change the path)

  .. code-block:: bash
 
   # add for spark
   export SPARK_HOME=your_spark_installation_path
   export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
   export PATH=$PATH:$SPARK_HOME/bin
   export PYSPARK_DRIVE_PYTHON="jupyter"
   export PYSPARK_DRIVE_PYTHON_OPTS="notebook"   

  At last, remember to source your ``bashrc``
   
  .. code-block:: bash
 
   source ~/.bashrc

Configure Spark on Windows
++++++++++++++++++++++++++

Installing open source software on Windows is always a nightmare for me. 
Thanks for Deelesh Mandloi. You can follow the detailed procedures in the 
blog `Getting Started with PySpark on Windows`_ to install the Apache Spark™
on your Windows Operator System.     

PySpark With Text Editor or IDE
+++++++++++++++++++++++++++++++

PySpark With Jupyter Notebook
-----------------------------

After you finishing the above setup steps in :ref:`set-up-Ubuntu`, 
then you should be good to write and run your PySpark Code
in Jupyter notebook.

  .. _fig_jupyterWithPySpark:
  .. figure:: images/jupyterWithPySpark.png
    :align: center   

PySpark With PyCharm
--------------------

After you finishing the above setup steps in :ref:`set-up-Ubuntu`, 
then you should be good to add the PySpark to your PyCharm project.

1. Create a new PyCharm project

  .. figure:: images/new_project.png
    :align: center   

2. Go to Project Structure
   
   Option 1: File -> Settings -> Project: -> Project Structure
   
   Option 2: PyCharm -> Preferences -> Project: -> Project Structure

  .. figure:: images/projectStructure.png
    :align: center   

3. Add Content Root: all ``ZIP`` files from $SPARK_HOME/python/lib

  .. figure:: images/add_root.png
    :align: center   

  .. figure:: images/added_root.png
    :align: center   

4. Run your script 

  .. figure:: images/run_test.png
    :align: center   


PySpark With Apache Zeppelin
----------------------------

After you finishing the above setup steps in :ref:`set-up-Ubuntu`, 
then you should be good to write and run your PySpark Code
in Apache Zeppelin.

  .. _fig_zeppelin:
  .. figure:: images/zeppelin.png
    :align: center   


PySpark With Sublime Text
-------------------------
 
After you finishing the above setup steps in :ref:`set-up-Ubuntu`, 
then you should be good to use Sublime Text to write your PySpark 
Code and run your code as a normal python code in Terminal.

 .. code-block:: bash
    
    python test_pyspark.py

Then you should get the output results in your terminal.  

  .. _fig_sublimeWithPySpark:
  .. figure:: images/sublimeWithPySpark.png
    :align: center



PySpark With Eclipse
--------------------

If you want to run PySpark code on Eclipse, you need to add the 
paths for the **External Libraries** for your **Current Project**
as follows:

 1. Open the properties of your project

  .. _fig_PyDevProperties:
  .. figure:: images/PyDevProperties.png
    :align: center

 2. Add the paths for the **External Libraries**

  .. _fig_pydevPath:
  .. figure:: images/pydevPath.png
    :align: center
 
And then you should be good to run your code on Eclipse with PyDev. 

  .. _fig_pysparkWithEclipse:
  .. figure:: images/pysparkWithEclipse.png
    :align: center    

.. index:: Set up Spark on Cloud


PySparkling Water: Spark + H2O
++++++++++++++++++++++++++++++

1. Download ``Sparkling Water`` from: https://s3.amazonaws.com/h2o-release/sparkling-water/rel-2.4/5/index.html

2. Test PySparking

.. code-block:: bash
 
  unzip sparkling-water-2.4.5.zip 
  cd  ~/sparkling-water-2.4.5/bin
  ./pysparkling

If you have a correct setup for PySpark, then you will get the following results:

.. code-block:: bash

   Using Spark defined in the SPARK_HOME=/Users/dt216661/spark environmental property
 
   Python 3.7.1 (default, Dec 14 2018, 13:28:58)
   [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   2019-02-15 14:08:30 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
   Setting default log level to "WARN".
   Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
   Setting default log level to "WARN".
   To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
   2019-02-15 14:08:31 WARN  Utils:66 - Service 'SparkUI' could not bind on port 4040. Attempting port 4041. 
   2019-02-15 14:08:31 WARN  Utils:66 - Service 'SparkUI' could not bind on port 4041. Attempting port 4042.
   17/08/30 13:30:12 WARN NativeCodeLoader: Unable to load native-hadoop 
   library for your platform... using builtin-java classes where applicable
   17/08/30 13:30:17 WARN ObjectStore: Failed to get database global_temp, 
   returning NoSuchObjectException 
   Welcome to
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version 2.4.0
          /_/

   Using Python version 3.7.1 (default, Dec 14 2018 13:28:58)
   SparkSession available as 'spark'.

3. Setup ``pysparkling`` with Jupyter notebook 

Add the following alias to your ``bashrc`` (Linux systems) or ``bash_profile`` (Mac system)

.. code-block:: bash

	alias sparkling="PYSPARK_DRIVER_PYTHON="ipython" PYSPARK_DRIVER_PYTHON_OPTS=    "notebook" /~/~/sparkling-water-2.4.5/bin/pysparkling"

4. Open ``pysparkling`` in terminal 

.. code-block:: bash

	sparkling	  

Set up Spark on Cloud
+++++++++++++++++++++
 
Following the setup steps in :ref:`set-up-Ubuntu`, you can set 
up your own cluster on the cloud, for example AWS, Google Cloud.
Actually, for those clouds, they have their own Big Data tool.
You can run them directly whitout any setting just like 
Databricks Community Cloud. If you want more details, please feel 
free to contact with me.

PySpark on Colaboratory
+++++++++++++++++++++++

Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud.


Installation
------------

.. code-block:: bash

	!pip install pyspark

Testing
-------

.. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark create RDD example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()
		
	df = spark.sparkContext\
	          .parallelize([(1, 2, 3, 'a b c'),
	                        (4, 5, 6, 'd e f'),
	                        (7, 8, 9, 'g h i')])\
	          .toDF(['col1', 'col2', 'col3','col4'])

	df.show()


Output:

.. code-block:: python

	+----+----+----+-----+
	|col1|col2|col3| col4|
	+----+----+----+-----+
	|   1|   2|   3|a b c|
	|   4|   5|   6|d e f|
	|   7|   8|   9|g h i|
	+----+----+----+-----+


Demo Code in this Section
+++++++++++++++++++++++++

The  Jupyter notebook can be download from `installation on colab <https://colab.research.google.com/drive/15LvijFl1gFoazvWlPxGFbYUl43KubrW1#scrollTo=mGHjEx_yixDx>`_.

* Python Source code

 .. literalinclude:: /code/test_pyspark.py


.. _Anaconda: https://www.anaconda.com/download/
.. _Java JDK: http://www.oracle.com/technetwork/java/javase/downloads/index-jsp-138363.html 
.. _How do I install Java for my Mac?: https://java.com/en/download/help/mac_install.xml
.. _Download Apache Spark™: http://spark.apache.org/downloads.html
.. _Getting Started with PySpark on Windows: http://deelesh.github.io/pyspark-windows.html



