/**
 *  @file      bogus_thread.cpp
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "bogus_thread.h"

namespace Hanami
{

BogusThread::BogusThread()
    : Hanami::Thread("BogusThread") {}

void BogusThread::run()
{
    while(m_abort == false) {
        sleepThread(10000);
    }
}

}
