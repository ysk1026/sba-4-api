import setuptools

with open("README.py", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='com_sba_api',
    version='1.0',
    description='Python Distribution Utilities',
    author='youngseonkim',
    author_email='gward@python.net',
    url='https://www.python.org/sigs/distutils-sig/',
    packages=['distutils', 'distutils.command'],
)