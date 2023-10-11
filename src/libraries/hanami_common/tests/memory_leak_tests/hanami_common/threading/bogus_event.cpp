/**
 *  @file      bogus_event.cpp
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "bogus_event.h"

namespace Hanami
{

BogusEvent::BogusEvent() {}

bool
BogusEvent::processEvent()
{
    return true;
}

}  // namespace Hanami
