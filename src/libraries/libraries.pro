TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

SUBDIRS =  hanami_common
SUBDIRS += hanami_crypto
SUBDIRS += hanami_ini
SUBDIRS += hanami_args
SUBDIRS += hanami_config
SUBDIRS += hanami_cpu
SUBDIRS += hanami_sqlite
# SUBDIRS += hanami_opencl
# SUBDIRS += hanami_obj
SUBDIRS += hanami_database
# SUBDIRS += hanami_network
SUBDIRS += hanami_policies
SUBDIRS += hanami_cluster_parser
SUBDIRS += hanami_files
SUBDIRS += hanami_hardware


hanami_crypto.depends = hanami_common
hanami_ini.depends = hanami_common
hanami_args.depends = hanami_common
hanami_config.depends = hanami_ini
hanami_cpu.depends = hanami_common
hanami_sqlite.depends = hanami_common
hanami_opencl.depends = hanami_common
#hanami_obj.depends = hanami_common
hanami_hardware.depends = hanami_cpu
hanami_database.depends = hanami_sqlite
#hanami_network.depends = libKitsunemimiNetwork
hanami_policies.depends = hanami_config hanami_args
libKitsunemimiHanamiSegmentParser.depends = hanami_config hanami_args
hanami_cluster_parser.depends = hanami_config hanami_args
src/hanami_sdk.depends = hanami_crypto
