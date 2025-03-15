from setuptools import setup,find_packages
def get_requirements(filename):
    with open(filename, 'r') as f:
        requires=f.readlines()
        requires=[x.replace('\n','') for x in requires]
        if '-e .' in requires:
            requires.remove('-e .')
    return requires

setup(
    name='Lyrics Predictor',
    version='1.0',
    packages=find_packages(),
    author='ayazulhaq yousafzai',
    author_email='syedthescientist@gmail.com',
    description='A lyrics predictor of songs using LSTM',
    install_requires=get_requirements('requirements.txt')
)

