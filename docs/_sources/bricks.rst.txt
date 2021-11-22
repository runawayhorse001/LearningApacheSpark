
.. _bricks:

===============
Databricks Tips
===============

In this chapter, I will share some of the useful tips when using Databricks.

Display samples
+++++++++++++++

In pyspark, we can use ``show()`` to display the given size of the sample. While in databricks environment, we also
have ``display()`` function to display the sample records. The CPU time for big data table/set is

.. code-block:: python

        display(df) < df.limit(n).show() < df.show(n) # n is the number of the given size

Auto files download
+++++++++++++++++++

Databricks is the most powerful big data analytics and machine learning platform, while it's not perfect. The file
management system is not that good like jupyter Notebook/Lab. Here I will provide one way to download the files under
``dbfs:/FileStore`` (This method only works for the files under ``dbfs:/FileStore``).

In general, the file link for downloading is like:

.. code-block:: python

    f"{cluster_address}/files/{file_path}?o={cluster_no}"


Here I provided my auto click download functions:

.. code-block:: python

    import os
    import IPython
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from jinja2 import Template
    from pathlib import Path
    from subprocess import Popen, PIPE

    def get_hdfs_files(hdfs_path, relative_path=True):
    """
    Get file names and file path or relative path for the given HDFS path. Note: os.listdir does not work in
    `community.cloud.databricks`.

    :param hdfs_path: the input HDFS path
    :param relative_path: flag of return full path or the path relative to ``dbfs:/FileStore`` (We need the relative
                          for the file download.)

    :return file_names: file names under the given path
    :return  relative_p: file paths, full path if ``relative_path=False`` else paths relative to ``bdfs:/FileStore``
    """
    # get the file information
    xx = dbutils.fs.ls(hdfs_path)

    # get hdfs path and folder name
    file_names = [list(xx[i])[1] for i in range(len(xx))]
    hdfs_paths = [list(xx[i])[0] for i in range(len(xx))]

    if relative_path:
        try:
            relative_p = [os.path.relpath(hdfs_path, 'dbfs:/FileStore') for hdfs_path in hdfs_paths]
        except:
            print("Only suooprt the files under 'dbfs:/FileStore/'")
    else:
        relative_p = hdfs_paths

    return file_names, relative_p


    def azdb_files_download(files_path, cluster_address="https://community.cloud.databricks.com",
                        cluster_no='4622560542654492'):
    """
    List the files download links.

    :param files_path: the given file path ot folder path
    :param cluster_address: Your databricks cluster address, i.e. the link before ``/?o``
    :param cluster_no: YOur databricks cluster number, i.e. the number after ``?o=``
    """

    if not os.path.isfile(files_path):  # os.path.isdir(files_path):
        file_names, files_path = get_hdfs_files(files_path)

    if not isinstance(files_path, list):
        files_path = [files_path]

    urls = [f"{cluster_address}/files/{file_path}?o={cluster_no}" for file_path in files_path]

    temp = """
           <h2>AZDB files download</h2>
           {% for i in range(len(urls)) %}

              <a href="{{urls[i]}}" target='_blank'> Click me to download  {{files_path[i].split('/')[-1]}}</a> <br>
           {% endfor %}
           """

    html = Template(temp).render(files_path=files_path, urls=urls, len=len)

    # get dbutils module
    dbutils = IPython.get_ipython().user_ns["dbutils"]

    dbutils.displayHTML(html)

.. note::

    In commercial version of databricks, you can use

    .. code-block:: python

          spark.conf.get("spark.databricks.clusterUsageTags.instanceWorkerEnvId")

    to get the ``cluster_no``. But it will not work for community version.


By using the above code, you can download the files relative to ``dbfs:/FileStore``.

The files under ``dbfs:/FileStore/data``

  .. _dbfs_data:
  .. figure:: images/dbfs_data.png
    :align: center

    File under ``dbfs:/FileStore/data``

Click download demos:

  .. _auto_download:
  .. figure:: images/auto_download.png
    :align: center

    File download in databricks

``delta`` format
++++++++++++++++

TODO...

``mlflow``
++++++++++

TODO...