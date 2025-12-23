#!/usr/bin/env python

from distutils.core import setup

package_name='vikit_py'

setup(
    name=package_name,
    version='0.0.1',
    install_requires=['setuptools'],
    zip_safe=True,
    packages=['vikit_py'],
    package_dir={'': 'src'},
    maintainer='Christian Forster',
    maintainer_email='forster@ifi.uzh.ch',
    description='The vikit_py package',
    license='BSD',
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
)
