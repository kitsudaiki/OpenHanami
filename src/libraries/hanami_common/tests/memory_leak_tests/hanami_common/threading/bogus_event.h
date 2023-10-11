/**
 *  @file      bogus_event.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef BOGUS_EVENT_H
#define BOGUS_EVENT_H

#include <hanami_common/threading/event.h>

namespace Hanami
{

class BogusEvent : public Hanami::Event
{
   public:
    BogusEvent();

    bool processEvent();
};

}  // namespace Hanami

#endif  // BOGUS_EVENT_H
