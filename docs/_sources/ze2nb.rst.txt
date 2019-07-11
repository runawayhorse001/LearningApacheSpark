
.. _ze2nb:

.. |eg| replace:: For example:

.. |re| replace:: Result:

============================
Zeppelin to jupyter notebook
============================

The Zeppelin users may have same problem with me that the Zeppelin ``.json`` notebook is hard to open and read. **ze2nb**: A piece of code to convert Zeppelin ``.json`` notebook to ``.ipynb`` Jupyter notebook, ``.py`` and ``.html`` file. This library is based on Ryan Blue's Jupyter/Zeppelin conversion: [jupyter-zeppelin]_. The API book can be found at `ze2nb API`_ or [zeppelin2nb]_.  **You may download and distribute it. Please be aware, however, that the note contains typos as well as inaccurate or incorrect description.** 


How to Install
++++++++++++++

.. |rst| replace:: ``Results``:
..
	.. admonition:: Chinese proverb

	   If you only know yourself, but not your opponent, you may win or may lose.
	   If you know neither yourself nor your enemy, you will always endanger yourself. 
	   – idiom, from Sunzi’s Art of War

Install with ``pip``
--------------------

You can install the ``ze2nb`` from [PyPI](https://pypi.org/project/ze2nb):

.. code-block:: bash

    pip install ze2nb


Install from Repo
-----------------


1. Clone the Repository


.. code-block:: bash

	git clone https://github.com/runawayhorse001/ze2nb.git


2. Install


.. code-block:: bash

	cd zeppelin2nb
	pip install -r requirements.txt 
	python setup.py install

Uninstall
---------

.. code-block:: bash

	pip uninstall ze2nb

Converting Demos
++++++++++++++++

The following demos are designed to show how to use ``zepplin2nb`` to convert the ``.json`` to  ``.ipynb`` , ``.py`` and ``.html``. 

Converting in one function 
--------------------------

|eg|

.. literalinclude:: code/demo_ze2nb.py
     :language: python


Converted results 
-----------------

|re|     
 
.. figure:: images/test.png
	:align: center

Results in output:

.. figure:: images/output.png
	:align: center

Results in output1:

.. figure:: images/output1.png
	:align: center    


.. _ze2nb API: https://runawayhorse001.github.io/ze2nb/