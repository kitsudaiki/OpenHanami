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

namespace Kitsunemimi
{

class BogusEvent
        : public Kitsunemimi::Event
{
public:
    BogusEvent();

    bool processEvent();
};

}

#endif // BOGUS_EVENT_H
