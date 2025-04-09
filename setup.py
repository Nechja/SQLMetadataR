from setuptools import setup, find_packages

setup(
    name='SQLMetadataR',
    version='0.1.01',
    description='A tool for extracting and analyzing metadata from SQLite databases.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Nechja/SQLMetadataR',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g., 'numpy', 'pandas'
    ],
    license='Attribution-ShareAlike 4.0 International License',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)