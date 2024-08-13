from setuptools import setup, find_packages


setup(
    name='memory-gelato',
    version='0.0.1',
    description='LLM fine-tuning peak memory estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/raul-singh/memory-gelato',
    author='Raul Singh',
    author_email='raul.singh@mail.polimi.it',
    license='BSD 3-Clause',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)