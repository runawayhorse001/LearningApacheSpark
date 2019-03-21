# Learning Apache Spark 

Website: https://runawayhorse001.github.io/LearningApacheSpark/


This is a shared repository for Learning Apache Spark Notes. The first version was posted on Github in [Feng2017]. This shared repository mainly contains the self-learning and self-teaching notes from Wenqiang during his IMA Data Science Fellowship.

In this repository, I try to use the detailed demo code and examples to show how to use each main functions. If you find your work wasn’t cited in this note, please feel free to let me know.

Although I am by no means an data mining programming and Big Data expert, I decided that it would be useful for me to share what I learned about PySpark programming in the form of easy tutorials with detailed example. I hope those tutorials will be a valuable tool for your studies.

The tutorials assume that the reader has a preliminary knowledge of programing and Linux. And this document is generated automatically by using sphinx.


BTW, I successfully brought git output format into Sphnix in this repository.  You need to install `sphinx-to-github`
and more details can be found from the following reference:

Reference:

- https://github.com/michaeljones/sphinx-to-github

Now, the ``sphinx-to-github`` function for github pages can be easily solved by add an empty file ``.nojekyll `` to your docs folder.  I add the following piece of code in my ``docgen.py`` to add it automatically: 


```
    # add .nojekyll file to fix the github pages issues
    nojekyll_path = os.path.join(outdir, '.nojekyll')
    if not os.path.exists(nojekyll_path):
        nojekyll = open(nojekyll_path,'a')
        nojekyll.close()
```    