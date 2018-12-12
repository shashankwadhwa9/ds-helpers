import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ds_helpers',
    version='0.1.1',
    author='Shashank Wadhwa',
    author_email='shashankwadhwa9@gmail.com',
    description='Helper functions for Data analysis and modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shashankwadhwa9/ds-helpers',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
