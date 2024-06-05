from setuptools import setup, find_packages

setup(
    name='Synopsys_Data_Challenge',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'librosa',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'run_job=Synopsys_Data_Challenge.main:run_job',
        ],
    },
    author='Ritik',
    author_email='rkhandelwal1511@gmail.com',
    description='A package to classify sounds using SVM and Random Forest',
    url='https://github.com/ritikhandelwal/Data_Challenge_Synopsys',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
