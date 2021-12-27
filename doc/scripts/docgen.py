#############################################################################
# I heavily borrowed, modified and used the configuration in docgen.py of Theano
# package project. I will keep all the comments from Theano team and the 
# coryright of this file belongs to Theano team. 
# reference: 
#           
# Theano repository: https://github.com/Theano/Theano
# docgen.py: https://github.com/Theano/Theano/blob/master/doc/scripts/docgen.py
##############################################################################
from __future__ import print_function
import sys
import os
import shutil
import inspect
import getopt
from collections import defaultdict

if __name__ == '__main__':

    throot = os.path.abspath(
        os.path.join(sys.path[0], os.pardir, os.pardir))

    options = defaultdict(bool)
    opts, args = getopt.getopt(
        sys.argv[1:],
        'o:f:',
        ['rst', 'help', 'nopdf', 'cache', 'check', 'test'])
    options.update(dict([x, y or True] for x, y in opts))
    if options['--help']:
        print('Usage: %s [OPTIONS] [files...]' % sys.argv[0])
        print('  -o <dir>: output the html files in the specified dir')
        print('  --cache: use the doctree cache')
        print('  --rst: only compile the doc (requires sphinx)')
        print('  --nopdf: do not produce a PDF file from the doc, only HTML')
        print('  --test: run all the code samples in the documentaton')
        print('  --check: treat warnings as errors')
        print('  --help: this help')
        print('If one or more files are specified after the options then only '
              'those files will be built. Otherwise the whole tree is '
              'processed. Specifying files will implies --cache.')
        sys.exit(0)

    if not(options['--rst'] or options['--test']):
        # Default is now rst
        options['--rst'] = True

    def mkdir(path):
        try:
            os.mkdir(path)
        except OSError:
            pass

    # create the putput folder docs, since github page will use /docs folder for Github page.
    outdir = options['-o'] or (throot + '/docs')
    # create the output folder latex
    latexdir = options['-o'] or (throot + '/latex')

    files = None
    if len(args) != 0:
        files = [os.path.abspath(f) for f in args]
    currentdir = os.getcwd()
    mkdir(outdir)
    #mkdir(latexdir)
    os.chdir(outdir)

    # add .gitignore file to your github repository
    ignore_path = os.path.join(throot, '.gitignore')
    if not os.path.exists(ignore_path):
        ignore_txt = open(throot + '/doc/scripts/'+'gitignore.txt')  
        gitignore = open(ignore_path,'a')
        for x in ignore_txt.readlines():
            gitignore.write(x)
        ignore_txt.close()
        gitignore.close()

    # add .nojekyll file to fix the github pages issues
    nojekyll_path = os.path.join(outdir, '.nojekyll')
    if not os.path.exists(nojekyll_path):
        nojekyll = open(nojekyll_path,'a')
        nojekyll.close()

    # Make sure the appropriate 'theano' directory is in the PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath = os.pathsep.join([throot, pythonpath])
    sys.path[0:0] = [throot]  # We must not use os.environ.

    # Make sure we don't use gpu to compile documentation
    env_th_flags = os.environ.get('THEANO_FLAGS', '')
    os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'
    
   

    def call_sphinx(builder, workdir):
        import sphinx
        if options['--check']:
            extraopts = ['-W']
        else:
            extraopts = []
        if not options['--cache'] and files is None:
            extraopts.append('-E')
        docpath = os.path.join(throot, 'doc')
        inopt = [docpath, workdir]
        if files is not None:
            inopt.extend(files)
        try:
            import sphinx.cmd.build
            ret = sphinx.cmd.build.build_main(
                ['-b', builder] + extraopts + inopt)
        except ImportError:
            # Sphinx < 1.7 - build_main drops first argument
            ret = sphinx.build_main(
                ['', '-b', builder] + extraopts + inopt)
        if ret != 0:
            sys.exit(ret)

    if options['--all'] or options['--rst']:
        mkdir("doc")
        sys.path[0:0] = [os.path.join(throot, 'doc')]
        call_sphinx('html', '.')

        if not options['--nopdf']:
            # Generate latex file in a temp directory
            import tempfile
            workdir = tempfile.mkdtemp()
            #workdir = latexdir
            call_sphinx('latex', workdir)
            # Compile to PDF
            os.chdir(workdir)
            os.system('make')
            try:
                shutil.copy(os.path.join(workdir, 'pyspark.pdf'), outdir)
                os.chdir(outdir)
                # remove the workdir folder
                shutil.rmtree(workdir)
            except OSError as e:
                print('OSError:', e)
            except IOError as e:
                print('IOError:', e)

    if options['--test']:
        mkdir("doc")
        sys.path[0:0] = [os.path.join(throot, 'doc')]
        call_sphinx('doctest', '.')

    # To go back to the original current directory.
    os.chdir(currentdir)



