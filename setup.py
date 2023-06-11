import setuptools
from setuptools.command.install import install
import subprocess


class NLTKDownloadCommand(install):
    def run(self):
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('maxent_ne_chunker')
        install.run(self)


setuptools.setup(
    # Other setup configurations...
    cmdclass={
        '': NLTKDownloadCommand,
    },
)
