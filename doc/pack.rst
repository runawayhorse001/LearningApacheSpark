
.. _pack:

====================
Wrap PySpark Package
====================

It's super easy to wrap your own package in Python. I packed some functions which I frequently 
used in my daily work. You can download and install it from `My PySpark Package`_. The hierarchical 
structure and the directory structure of this package are as follows. 
 

Package Wrapper 
+++++++++++++++

Hierarchical Structure
----------------------


.. code-block:: bash

	|-- build
	|   |-- bdist.linux-x86_64
	|   |-- lib.linux-x86_64-2.7
	|       |-- PySparkTools
	|           |-- __init__.py
	|           |-- Manipulation
	|           |   |-- DataManipulation.py
	|           |   |-- __init__.py
	|           |── Visualization
	|               |-- __init__.py
	│               |-- PyPlots.py
	|-- dist
	│   |-- PySParkTools-1.0-py2.7.egg
	|-- __init__.py
	|-- PySparkTools
	|   |-- __init__.py
	|   |-- Manipulation
	|   |   |-- DataManipulation.py
	|   |   |-- __init__.py
	|   |-- Visualization
	|       |-- __init__.py
	|       |-- PyPlots.py
	│       |-- PyPlots.pyc
	|-- PySParkTools.egg-info
	|   |-- dependency_links.txt
	|   |-- PKG-INFO
	|   |-- requires.txt
	|   |-- SOURCES.txt
	|   |-- top_level.txt
	|-- README.md
	|-- requirements.txt
	|-- setup.py
	|-- test
	    |-- spark-warehouse
	    |-- test1.py
	    |-- test2.py



From the above hierarchical structure, you will find that you have to have ``__init__.py`` in each directory. I will explain the ``__init__.py`` file with the example below:

Set Up
------

.. code-block:: python

	from setuptools import setup, find_packages

	try:
	    with open("README.md") as f:
	        long_description = f.read()
	except IOError:
	    long_description = ""

	try:
	    with open("requirements.txt") as f:
	        requirements = [x.strip() for x in f.read().splitlines() if x.strip()]
	except IOError:
	    requirements = []

	setup(name='PySParkTools',
		  install_requires=requirements,
	      version='1.0',
	      description='Python Spark Tools',
	      author='Wenqiang Feng',
	      author_email='von198@gmail.com',
	      url='https://github.com/runawayhorse001/PySparkTools',
	      packages=find_packages(),
	      long_description=long_description
	     )

ReadMe
------

.. code-block:: bash

	# PySparkTools

	This is my PySpark Tools. If you want to colne and install it, you can use 

	- clone

	```{bash}
	git clone git@github.com:runawayhorse001/PySparkTools.git
	```
	- install 

	```{bash}
	cd PySparkTools
	pip install -r requirements.txt 
	python setup.py install
	```

	- test 

	```{bash}
	cd PySparkTools/test
	python test1.py
	```

Pacakge Publishing on PyPI
++++++++++++++++++++++++++

Install ``twine``
-----------------

.. code-block:: bash

	pip install twine

Build Your Package
------------------

.. code-block:: python

	python setup.py sdist bdist_wheel

Then you will get a new folder ``dist``:

.. code-block:: bash

	.
	├── PySparkAudit-1.0.0-py2.7.egg
	├── PySparkAudit-1.0.0-py2-none-any.whl
	└── PySparkAudit-1.0.0.tar.gz


Upload Your Package
-------------------

.. code-block:: bash

	twine upload dist/*

During the uploading processing, you need to provide your PyPI account ``username`` and ``password``:

.. code-block:: bash

	Enter your username: runawayhorse001
	Enter your password: ***************

Package at PyPI
---------------

Here is my ``PySparkAudit`` package at [PyPI](https://pypi.org/project/PySparkAudit). You can install ``PySparkAudit`` using:

.. code-block:: bash

    pip install PySparkAudit


.. _My PySpark Package: https://github.com/runawayhorse001/PySparkAudit
