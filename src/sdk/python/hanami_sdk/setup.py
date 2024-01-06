from setuptools import setup
from setuptools.command.install import install
from subprocess import check_call

class GenerateProtobufMessages(install):
    def run(self):
        # Run your custom command here
        check_call(["protoc",
                    "--python_out=./hanami_sdk",
                    "--proto_path",
                    "../../../libraries/hanami_messages/protobuffers",
                    "hanami_messages.proto3"])

        # Continue with the default installation process
        install.run(self)

setup(
    name='hanami_sdk',
    version='0.4.0b0',
    description='SDK library for Hanami',
    url='https://github.com/kitsudaiki/Hanami',
    author='Tobias Anker',
    author_email='tobias.anker@kitsunemimi.moe',
    license='Apache 2.0',
    packages=['hanami_sdk', 'hanami_sdk.hanami_messages'],
    install_requires=['jsonschema==3.2.0',
                      'protobuf==3.19.6',
                      'requests==2.31.0',
                      'simplejson==3.17.6',
                      'websockets==12.0'],
    cmdclass={
        'install': GenerateProtobufMessages,
    },
    classifiers=[
        'License :: Apache 2.0',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
