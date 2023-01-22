TEMPLATE = subdirs
CONFIG += ordered
QT -= qt core gui
CONFIG += c++17

libHanamiAiSdk.file = libHanamiAiSdk/cpp/libHanamiAiSdk.pro

SUBDIRS = libKitsunemimiCommon
SUBDIRS += libKitsunemimiNetwork
SUBDIRS += libKitsunemimiJson
SUBDIRS += libKitsunemimiCrypto
SUBDIRS += libKitsunemimiJwt
SUBDIRS += libKitsunemimiIni
SUBDIRS += libKitsunemimiArgs
SUBDIRS += libKitsunemimiConfig
SUBDIRS += libKitsunemimiCpu
SUBDIRS += libKitsunemimiSqlite
SUBDIRS += libKitsunemimiOpencl
SUBDIRS += libKitsunemimiSakuraHardware
SUBDIRS += libKitsunemimiSakuraDatabase
SUBDIRS += libKitsunemimiSakuraNetwork
SUBDIRS += libKitsunemimiHanamiCommon
SUBDIRS += libKitsunemimiHanamiPolicies
SUBDIRS += libKitsunemimiHanamiDatabase
SUBDIRS += libKitsunemimiHanamiSegmentParser
SUBDIRS += libKitsunemimiHanamiClusterParser
SUBDIRS += libKitsunemimiHanamiNetwork
SUBDIRS += libHanamiAiSdk
SUBDIRS += libShioriArchive
SUBDIRS += libMisakiGuard
SUBDIRS += libAzukiHeart
SUBDIRS += ToriiGateway
SUBDIRS += ShioriArchive
SUBDIRS += MisakiGuard
SUBDIRS += AzukiHeart
SUBDIRS += KyoukoMind
SUBDIRS += TsugumiTester

libKitsunemimiNetwork.depends = libKitsunemimiCommon
libKitsunemimiJson.depends = libKitsunemimiCommon
libKitsunemimiCrypto.depends = libKitsunemimiCommon
libKitsunemimiJwt.depends = libKitsunemimiCrypto libKitsunemimiJson
libKitsunemimiIni.depends = libKitsunemimiCommon
libKitsunemimiArgs.depends = libKitsunemimiCommon
libKitsunemimiConfig.depends = libKitsunemimiIni
libKitsunemimiCpu.depends = libKitsunemimiCommon
libKitsunemimiSqlite.depends = libKitsunemimiCommon
libKitsunemimiOpencl.depends = libKitsunemimiCommon
libKitsunemimiSakuraHardware.depends = libKitsunemimiCpu
libKitsunemimiSakuraDatabase.depends = libKitsunemimiSqlite
libKitsunemimiSakuraNetwork.depends = libKitsunemimiNetwork
libKitsunemimiHanamiCommon.depends = libKitsunemimiConfig libKitsunemimiArgs
libKitsunemimiHanamiPolicies.depends = libKitsunemimiConfig libKitsunemimiArgs
libKitsunemimiHanamiDatabase.depends = libKitsunemimiJson libKitsunemimiSakuraDatabase
libKitsunemimiHanamiSegmentParser.depends = libKitsunemimiConfig libKitsunemimiArgs libKitsunemimiHanamiCommon
libKitsunemimiHanamiClusterParser.depends = libKitsunemimiConfig libKitsunemimiArgs libKitsunemimiHanamiCommon
libKitsunemimiHanamiNetwork.depends = libKitsunemimiHanamiCommon libKitsunemimiCrypto libKitsunemimiJson
libHanamiAiSdk.depends = libKitsunemimiHanamiCommon libKitsunemimiCrypto libKitsunemimiJson 
libShioriArchive.depends = libKitsunemimiHanamiNetwork
libMisakiGuard.depends = libKitsunemimiHanamiNetwork
libAzukiHeart.depends = libKitsunemimiHanamiNetwork
ToriiGateway.depends = libKitsunemimiHanamiNetwork
ShioriArchive.depends = libKitsunemimiHanamiNetwork libKitsunemimiHanamiDatabase
MisakiGuard.depends = libKitsunemimiHanamiNetwork libKitsunemimiHanamiDatabase libKitsunemimiHanamiPolicies
AzukiHeart.depends = libKitsunemimiHanamiNetwork libKitsunemimiHanamiDatabase
KyoukoMind.depends = libKitsunemimiHanamiNetwork libKitsunemimiOpencl libKitsunemimiHanamiClusterParser libKitsunemimiHanamiSegmentParser libKitsunemimiHanamiDatabase libHanamiAiSdk
TsugumiTester.depends = libKitsunemimiHanamiNetwork ibKitsunemimiHanamiClusterParser libKitsunemimiHanamiSegmentParser libKitsunemimiHanamiDatabase libHanamiAiSdk





