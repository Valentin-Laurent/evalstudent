from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='evalstudent',
      version="1.0",
      description="Exploratory notebooks and utils related to the Feedback Prize - Evaluating Student Writing Kaggle competition.",
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
