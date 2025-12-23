from setuptools import setup
from glob import glob


conda_python = '/home/unitree/miniconda3/envs/mapdsg/bin/python'
package_name = 'goal_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yu.zhao',
    maintainer_email='yu.zhao@horizon.auto',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'goal_pose_publisher = goal_publisher.goal_pose_publisher:main',
        ],
    },
    options={'build_scripts': {'executable': conda_python}},
)
