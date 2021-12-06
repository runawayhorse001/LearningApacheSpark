# -*- coding: utf-8 -*-
#############################################################################
# I heavily borrowed, modified and used the configuration in conf.py of Theano
# package project. I will keep all the comments from Theano team and the 
# coryright of this file belongs to Theano team. 
# reference: 
#           
# Theano repository: https://github.com/Theano/Theano
# conf.py: https://github.com/Theano/Theano/blob/master/doc/conf.py 
##############################################################################
# theano documentation build configuration file, created by
# sphinx-quickstart on Tue Oct  7 16:34:06 2008.
#
# This file is execfile()d with the current directory set to its containing
# directory.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed
# automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#sys.path.append(os.path.abspath('some/directory'))

from __future__ import absolute_import, print_function, division

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import versioneer

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.todo',
              'sphinx.ext.doctest',
              'sphinx.ext.napoleon',
              'sphinx.ext.linkcode',
              'sphinx.ext.intersphinx' 
              ]

todo_include_todos = True
napoleon_google_docstring = False
napoleon_include_special_with_doc = False


# We do it like this to support multiple sphinx version without having warning.
# Our buildbot consider warning as error.
try:
    from sphinx.ext import imgmath
    extensions.append('sphinx.ext.imgmath')
except ImportError:
    try:
        from sphinx.ext import pngmath
        extensions.append('sphinx.ext.pngmath')
    except ImportError:
        pass


# Add any paths that contain templates here, relative to this directory.
#templates_path = ['.templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General substitutions.
project = 'Learning Apache Spark with Python'
copyright = '2017, Wenqiang Feng'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#

# We need this hokey-pokey because versioneer needs the current
# directory to be the root of the project to work.
# The short X.Y version.
# version = '1.00'
# The full version, including alpha/beta/rc tags.
# release = '1.00'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directories, that shouldn't be
# searched for source files.
exclude_dirs = ['images', 'scripts', 'sandbox']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Options for HTML output
# -----------------------

# Enable link of 'View page source'
html_show_sourcelink = False
# Add 'Edit on Github' link instead of 'View page source'
# reference:https://docs.readthedocs.io/en/latest/vcs.html
# html_context = {
#     # Enable the "Edit in GitHub link within the header of each page.
#     'display_github': True,
#     # Set the following variables to generate the resulting github URL for each page. 
#     # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}
#     #/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
#     #https://github.com/runawayhorse001/SphinxGithub/blob/master/doc/index.rst
#     'github_user': 'runawayhorse001',
#     'github_repo': 'SphinxGithub',
#     'github_version': 'master/doc/' ,
# }

# {% if display_github %}
#     <li><a href="https://github.com/{{ github_user }}/{{ github_repo }}
#     /tree/{{ github_version }}{{ conf_py_path }}{{ pagename }}.rst">
#     Show on GitHub</a></li>
# {% endif %}

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'default.css'
# html_theme = 'sphinxdoc'

# Read the docs style:
if os.environ.get('READTHEDOCS') != 'True':
    try:
        import sphinx_rtd_theme
    except ImportError:
        pass  # assume we have sphinx >= 1.3
    else:
        html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme = 'sphinx_rtd_theme'

def setup(app):
    # app.add_stylesheet("fix_rtd.css")
    app.add_css_file("fix_rtd.css") 

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = 'images/theano_logo_allwhite_210x70.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = 'images/icon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'spnixgitdoc'

# Options for the linkcode extension
# ----------------------------------
# Resolve function
# This function is used to populate the (source) links in the API
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.abspath('..'))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = '%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    #https://github.com/runawayhorse001/LearningApacheSpark/blob/master/pyspark/ml/clustering.py
    return "https://github.com/runawayhorse001/LearningApacheSpark/blob/master/%s" % (filename)

# Options for LaTeX output
# ------------------------

latex_elements = {
    # The paper size ('letter' or 'a4').
    #latex_paper_size = 'a4',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    #latex_preamble = '',

    'figure_align': 'H',

}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class
# [howto/manual]).
latex_documents = [
  ('index', 'pyspark.tex', 'Learning Apache Spark with Python',
   'Wenqiang Feng', 'manual'),
]
# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = 'images/logo.jpg'
# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = 'images/snake_theta2-trans.png'
#latex_logo = 'images/theano_logo_allblue_200x46.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True


#latex_elements['preamble'] = '\usepackage{xcolor}'
# Additional stuff for the LaTeX preamble.
#latex_preamble 
latex_elements['preamble'] =  '\\usepackage{amsmath}\n'+\
                          '\\usepackage{mathtools}\n'+\
                          '\\usepackage{amsfonts}\n'+\
                          '\\usepackage{amssymb}\n'+\
                          '\\usepackage{dsfont}\n'+\
                          '\\def\\Z{\\mathbb{Z}}\n'+\
                          '\\def\\R{\\mathbb{R}}\n'+\
                          '\\def\\bX{\\mathbf{X}}\n'+\
                          '\\def\\X{\\mathbf{X}}\n'+\
                          '\\def\\By{\\mathbf{y}}\n'+\
                          '\\def\\Bbeta{{\\boldsymbol{\\beta}}}\n'+\
                          '\\def\\bU{\\mathbf{U}}\n'+\
                          '\\def\\bV{\\mathbf{V}}\n'+\
                          '\\def\\V1{\\mathds{1}}\n'+\
                          '\\def\\hU{\\mathbf{\hat{U}}}\n'+\
                          '\\def\\hS{\\mathbf{\hat{\Sigma}}}\n'+\
                          '\\def\\hV{\\mathbf{\hat{V}}}\n'+\
                          '\\def\\E{\\mathbf{E}}\n'+\
                          '\\def\\F{\\mathbf{F}}\n'+\
                          '\\def\\x{\\boldsymbol{x}}\n'+\
                          '\\def\\y{\\boldsymbol{y}}\n'+\
                          '\\def\\h{\\mathbf{h}}\n'+\
                          '\\def\\v{\\mathbf{v}}\n'+\
                          '\\def\\nv{\\mathbf{v^{{\bf -}}}}\n'+\
                          '\\def\\nh{\\mathbf{h^{{\bf -}}}}\n'+\
                          '\\def\\s{\\mathbf{s}}\n'+\
                          '\\def\\b{\\mathbf{b}}\n'+\
                          '\\def\\c{\\mathbf{c}}\n'+\
                          '\\def\\W{\\mathbf{W}}\n'+\
                          '\\def\\C{\\mathbf{C}}\n'+\
                          '\\def\\P{\\mathbf{P}}\n'+\
                          '\\def\\T{{\\bf \\mathcal T}}\n'+\
                          '\\def\\B{{\\bf \\mathcal B}}\n'

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

default_role = 'math'
pngmath_divpng_args = ['-gamma 1.5','-D 110']
#pngmath_divpng_args = ['-gamma', '1.5', '-D', '110', '-bg', 'Transparent'] 
imgmath_latex_preamble =  '\\usepackage{amsmath}\n'+\
                          '\\usepackage{mathtools}\n'+\
                          '\\usepackage{amsfonts}\n'+\
                          '\\usepackage{amssymb}\n'+\
                          '\\usepackage{dsfont}\n'+\
                          '\\def\\Z{\\mathbb{Z}}\n'+\
                          '\\def\\R{\\mathbb{R}}\n'+\
                          '\\def\\bX{\\mathbf{X}}\n'+\
                          '\\def\\X{\\mathbf{X}}\n'+\
                          '\\def\\By{\\mathbf{y}}\n'+\
                          '\\def\\Bbeta{{\\boldsymbol{\\beta}}}\n'+\
                          '\\def\\U{\\mathbf{U}}\n'+\
                          '\\def\\V{\\mathbf{V}}\n'+\
                          '\\def\\V1{\\mathds{1}}\n'+\
                          '\\def\\hU{\\mathbf{\hat{U}}}\n'+\
                          '\\def\\hS{\\mathbf{\hat{\Sigma}}}\n'+\
                          '\\def\\hV{\\mathbf{\hat{V}}}\n'+\
                          '\\def\\E{\\mathbf{E}}\n'+\
                          '\\def\\F{\\mathbf{F}}\n'+\
                          '\\def\\x{\\boldsymbol{x}}\n'+\
                          '\\def\\y{\\boldsymbol{y}}\n'+\
                          '\\def\\h{\\mathbf{h}}\n'+\
                          '\\def\\v{\\mathbf{v}}\n'+\
                          '\\def\\nv{\\mathbf{v^{{\bf -}}}}\n'+\
                          '\\def\\nh{\\mathbf{h^{{\bf -}}}}\n'+\
                          '\\def\\s{\\mathbf{s}}\n'+\
                          '\\def\\b{\\mathbf{b}}\n'+\
                          '\\def\\c{\\mathbf{c}}\n'+\
                          '\\def\\W{\\mathbf{W}}\n'+\
                          '\\def\\C{\\mathbf{C}}\n'+\
                          '\\def\\P{\\mathbf{P}}\n'+\
                          '\\def\\T{{\\bf \\mathcal T}}\n'+\
                          '\\def\\B{{\\bf \\mathcal B}}\n'

