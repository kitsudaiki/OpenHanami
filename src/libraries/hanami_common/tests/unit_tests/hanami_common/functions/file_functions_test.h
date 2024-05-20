/**
 *  @file      file_functions_test.h
 *
 *  @author    Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef FILE_functions_TEST_H
#define FILE_functions_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class Filefunctions_Test : public Hanami::CompareTestHelper
{
   public:
    Filefunctions_Test();

   private:
    void listFiles_test();
    void renameFileOrDir_test();
    void copyPath_test();
    void createDirectory_test();
    void deleteFileOrDir_test();
};

}  // namespace Hanami

#endif  // FILE_functions_TEST_H
