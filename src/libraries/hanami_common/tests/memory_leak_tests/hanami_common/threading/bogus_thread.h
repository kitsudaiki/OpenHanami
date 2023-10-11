/**
 *  @file      bogus_thread.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef BOGUS_THREAD_H
#define BOGUS_THREAD_H

#include <hanami_common/threading/thread.h>

namespace Hanami
{

class BogusThread : public Hanami::Thread
{
   public:
    BogusThread();

   protected:
    void run();
};

}  // namespace Hanami

#endif  // BOGUS_THREAD_H
