from setuptools import setup

conda_python = '/home/unitree/miniconda3/envs/mapdsg/bin/python'
package_name = 'chat_loc_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
       #
       # ('share/ament_index/resource_index/packages',
       #     ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
        'topic_chat_loc_pub = chat_loc_python.node_chat_loc_class:main'
        ],
    },
    options={'build_scripts': {'executable': conda_python}},
)
