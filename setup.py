from setuptools import setup, find_packages

setup(
    name = 'sneiaTools',
    version = '0.0.4',
    license = 'MIT',
    description = 'A python package that will include all the tools that the SNEIA research team will develop and need to carry out their tasks.',
    author = 'SNEIA Research Team',
    author_email = 'sneia@utp.edu.co',
    install_requires = [
    'numpy',
    'matplotlib',
    'pandas',
    'mne',
    'scipy',
    'scikit-learn'
    ],
    url = 'https://github.com/Semillero-de-neuroinformatica/SNEIA-tools',
    packages = find_packages()
)