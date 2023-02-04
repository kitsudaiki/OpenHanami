/**
 *  @file      bogus_thread.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef BOGUS_THREAD_H
#define BOGUS_THREAD_H

#include <libKitsunemimiCommon/threading/thread.h>

namespace Kitsunemimi
{

class BogusThread
        : public Kitsunemimi::Thread
{
public:
    BogusThread();

protected:
    void run();
};

}

#endif // BOGUS_THREAD_H
