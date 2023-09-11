/**
 * @file        direct_io_test.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#include "direct_io_test.h"

#include <hanami_sdk/io.h>
#include <common/test_thread.h>

DirectIoTest::DirectIoTest(const bool expectSuccess)
    : TestStep(expectSuccess)
{
    m_testName = "io-test";
    if(expectSuccess) {
      m_testName += " (success)";
    } else {
      m_testName += " (fail)";
    }
}

bool
DirectIoTest::runTest(json &,
                      Hanami::ErrorContainer &)
{
    if(trainTest() == false) {
        return false;
    }

    if(requestTest() == false) {
        return false;
    }

    return true;
}

bool
DirectIoTest::trainTest()
{
    Hanami::ErrorContainer error;

    // create input
    float inputValues[784];
    fillInputValues(&inputValues[0]);

    // create should
    float shouldValues[10];
    fillShouldValues(shouldValues);

    for(uint64_t i = 0; i < 100; i++)
    {
        std::cout<<"run: "<<i<<std::endl;

        if(Hanami::train(TestThread::m_wsClient,
                           inputValues,
                           784,
                           shouldValues,
                           10,
                           error) == false)
        {
            return false;
        }
    }

    return true;
}

bool
DirectIoTest::requestTest()
{
    Hanami::ErrorContainer error;

    // send request
    float inputValues[784];
    fillInputValues(&inputValues[0]);

    for(uint64_t i = 0; i < 784; i++) {
        inputValues[i] *= 255.0f;
    }

    uint64_t numberOfValues = 0;
    float* values = Hanami::request(TestThread::m_wsClient,
                                      inputValues,
                                      784,
                                      numberOfValues,
                                      error);
    if(values == nullptr) {
        return false;
    }

    std::cout<<"numberOfValues: "<<numberOfValues<<std::endl;
    for(uint32_t i = 0; i < numberOfValues; i++) {
        std::cout<<i<<": "<<values[i]<<std::endl;
    }

    delete[] values;

    return true;
}

void
DirectIoTest::fillShouldValues(float* shouldValues)
{
    shouldValues[0] = 0;
    shouldValues[1] = 0;
    shouldValues[2] = 0;
    shouldValues[3] = 0;
    shouldValues[4] = 0;
    shouldValues[5] = 1;
    shouldValues[6] = 0;
    shouldValues[7] = 0;
    shouldValues[8] = 0;
    shouldValues[9] = 0;
}

void
DirectIoTest::fillInputValues(float* inputValues)
{
    // fill a "5" into the input-buffer
    inputValues[0] = 0;
    inputValues[1] = 0;
    inputValues[2] = 0;
    inputValues[3] = 0;
    inputValues[4] = 0;
    inputValues[5] = 0;
    inputValues[6] = 0;
    inputValues[7] = 0;
    inputValues[8] = 0;
    inputValues[9] = 0;
    inputValues[10] = 0;
    inputValues[11] = 0;
    inputValues[12] = 0;
    inputValues[13] = 0;
    inputValues[14] = 0;
    inputValues[15] = 0;
    inputValues[16] = 0;
    inputValues[17] = 0;
    inputValues[18] = 0;
    inputValues[19] = 0;
    inputValues[20] = 0;
    inputValues[21] = 0;
    inputValues[22] = 0;
    inputValues[23] = 0;
    inputValues[24] = 0;
    inputValues[25] = 0;
    inputValues[26] = 0;
    inputValues[27] = 0;
    inputValues[28] = 0;
    inputValues[29] = 0;
    inputValues[30] = 0;
    inputValues[31] = 0;
    inputValues[32] = 0;
    inputValues[33] = 0;
    inputValues[34] = 0;
    inputValues[35] = 0;
    inputValues[36] = 0;
    inputValues[37] = 0;
    inputValues[38] = 0;
    inputValues[39] = 0;
    inputValues[40] = 0;
    inputValues[41] = 0;
    inputValues[42] = 0;
    inputValues[43] = 0;
    inputValues[44] = 0;
    inputValues[45] = 0;
    inputValues[46] = 0;
    inputValues[47] = 0;
    inputValues[48] = 0;
    inputValues[49] = 0;
    inputValues[50] = 0;
    inputValues[51] = 0;
    inputValues[52] = 0;
    inputValues[53] = 0;
    inputValues[54] = 0;
    inputValues[55] = 0;
    inputValues[56] = 0;
    inputValues[57] = 0;
    inputValues[58] = 0;
    inputValues[59] = 0;
    inputValues[60] = 0;
    inputValues[61] = 0;
    inputValues[62] = 0;
    inputValues[63] = 0;
    inputValues[64] = 0;
    inputValues[65] = 0;
    inputValues[66] = 0;
    inputValues[67] = 0;
    inputValues[68] = 0;
    inputValues[69] = 0;
    inputValues[70] = 0;
    inputValues[71] = 0;
    inputValues[72] = 0;
    inputValues[73] = 0;
    inputValues[74] = 0;
    inputValues[75] = 0;
    inputValues[76] = 0;
    inputValues[77] = 0;
    inputValues[78] = 0;
    inputValues[79] = 0;
    inputValues[80] = 0;
    inputValues[81] = 0;
    inputValues[82] = 0;
    inputValues[83] = 0;
    inputValues[84] = 0;
    inputValues[85] = 0;
    inputValues[86] = 0;
    inputValues[87] = 0;
    inputValues[88] = 0;
    inputValues[89] = 0;
    inputValues[90] = 0;
    inputValues[91] = 0;
    inputValues[92] = 0;
    inputValues[93] = 0;
    inputValues[94] = 0;
    inputValues[95] = 0;
    inputValues[96] = 0;
    inputValues[97] = 0;
    inputValues[98] = 0;
    inputValues[99] = 0;
    inputValues[100] = 0;
    inputValues[101] = 0;
    inputValues[102] = 0;
    inputValues[103] = 0;
    inputValues[104] = 0;
    inputValues[105] = 0;
    inputValues[106] = 0;
    inputValues[107] = 0;
    inputValues[108] = 0;
    inputValues[109] = 0;
    inputValues[110] = 0;
    inputValues[111] = 0;
    inputValues[112] = 0;
    inputValues[113] = 0;
    inputValues[114] = 0;
    inputValues[115] = 0;
    inputValues[116] = 0;
    inputValues[117] = 0;
    inputValues[118] = 0;
    inputValues[119] = 0;
    inputValues[120] = 0;
    inputValues[121] = 0;
    inputValues[122] = 0;
    inputValues[123] = 0;
    inputValues[124] = 0;
    inputValues[125] = 0;
    inputValues[126] = 0;
    inputValues[127] = 0;
    inputValues[128] = 0;
    inputValues[129] = 0;
    inputValues[130] = 0;
    inputValues[131] = 0;
    inputValues[132] = 0;
    inputValues[133] = 0;
    inputValues[134] = 0;
    inputValues[135] = 0;
    inputValues[136] = 0;
    inputValues[137] = 0;
    inputValues[138] = 0;
    inputValues[139] = 0;
    inputValues[140] = 0;
    inputValues[141] = 0;
    inputValues[142] = 0;
    inputValues[143] = 0;
    inputValues[144] = 0;
    inputValues[145] = 0;
    inputValues[146] = 0;
    inputValues[147] = 0;
    inputValues[148] = 0;
    inputValues[149] = 0;
    inputValues[150] = 0;
    inputValues[151] = 0;
    inputValues[152] = 0.0117647;
    inputValues[153] = 0.0705882;
    inputValues[154] = 0.0705882;
    inputValues[155] = 0.0705882;
    inputValues[156] = 0.494118;
    inputValues[157] = 0.533333;
    inputValues[158] = 0.686275;
    inputValues[159] = 0.101961;
    inputValues[160] = 0.65098;
    inputValues[161] = 1;
    inputValues[162] = 0.968627;
    inputValues[163] = 0.498039;
    inputValues[164] = 0;
    inputValues[165] = 0;
    inputValues[166] = 0;
    inputValues[167] = 0;
    inputValues[168] = 0;
    inputValues[169] = 0;
    inputValues[170] = 0;
    inputValues[171] = 0;
    inputValues[172] = 0;
    inputValues[173] = 0;
    inputValues[174] = 0;
    inputValues[175] = 0;
    inputValues[176] = 0.117647;
    inputValues[177] = 0.141176;
    inputValues[178] = 0.368627;
    inputValues[179] = 0.603922;
    inputValues[180] = 0.666667;
    inputValues[181] = 0.992157;
    inputValues[182] = 0.992157;
    inputValues[183] = 0.992157;
    inputValues[184] = 0.992157;
    inputValues[185] = 0.992157;
    inputValues[186] = 0.882353;
    inputValues[187] = 0.67451;
    inputValues[188] = 0.992157;
    inputValues[189] = 0.94902;
    inputValues[190] = 0.764706;
    inputValues[191] = 0.25098;
    inputValues[192] = 0;
    inputValues[193] = 0;
    inputValues[194] = 0;
    inputValues[195] = 0;
    inputValues[196] = 0;
    inputValues[197] = 0;
    inputValues[198] = 0;
    inputValues[199] = 0;
    inputValues[200] = 0;
    inputValues[201] = 0;
    inputValues[202] = 0;
    inputValues[203] = 0.192157;
    inputValues[204] = 0.933333;
    inputValues[205] = 0.992157;
    inputValues[206] = 0.992157;
    inputValues[207] = 0.992157;
    inputValues[208] = 0.992157;
    inputValues[209] = 0.992157;
    inputValues[210] = 0.992157;
    inputValues[211] = 0.992157;
    inputValues[212] = 0.992157;
    inputValues[213] = 0.984314;
    inputValues[214] = 0.364706;
    inputValues[215] = 0.321569;
    inputValues[216] = 0.321569;
    inputValues[217] = 0.219608;
    inputValues[218] = 0.152941;
    inputValues[219] = 0;
    inputValues[220] = 0;
    inputValues[221] = 0;
    inputValues[222] = 0;
    inputValues[223] = 0;
    inputValues[224] = 0;
    inputValues[225] = 0;
    inputValues[226] = 0;
    inputValues[227] = 0;
    inputValues[228] = 0;
    inputValues[229] = 0;
    inputValues[230] = 0;
    inputValues[231] = 0.0705882;
    inputValues[232] = 0.858824;
    inputValues[233] = 0.992157;
    inputValues[234] = 0.992157;
    inputValues[235] = 0.992157;
    inputValues[236] = 0.992157;
    inputValues[237] = 0.992157;
    inputValues[238] = 0.776471;
    inputValues[239] = 0.713726;
    inputValues[240] = 0.968627;
    inputValues[241] = 0.945098;
    inputValues[242] = 0;
    inputValues[243] = 0;
    inputValues[244] = 0;
    inputValues[245] = 0;
    inputValues[246] = 0;
    inputValues[247] = 0;
    inputValues[248] = 0;
    inputValues[249] = 0;
    inputValues[250] = 0;
    inputValues[251] = 0;
    inputValues[252] = 0;
    inputValues[253] = 0;
    inputValues[254] = 0;
    inputValues[255] = 0;
    inputValues[256] = 0;
    inputValues[257] = 0;
    inputValues[258] = 0;
    inputValues[259] = 0;
    inputValues[260] = 0.313726;
    inputValues[261] = 0.611765;
    inputValues[262] = 0.419608;
    inputValues[263] = 0.992157;
    inputValues[264] = 0.992157;
    inputValues[265] = 0.803922;
    inputValues[266] = 0.0431373;
    inputValues[267] = 0;
    inputValues[268] = 0.168627;
    inputValues[269] = 0.603922;
    inputValues[270] = 0;
    inputValues[271] = 0;
    inputValues[272] = 0;
    inputValues[273] = 0;
    inputValues[274] = 0;
    inputValues[275] = 0;
    inputValues[276] = 0;
    inputValues[277] = 0;
    inputValues[278] = 0;
    inputValues[279] = 0;
    inputValues[280] = 0;
    inputValues[281] = 0;
    inputValues[282] = 0;
    inputValues[283] = 0;
    inputValues[284] = 0;
    inputValues[285] = 0;
    inputValues[286] = 0;
    inputValues[287] = 0;
    inputValues[288] = 0;
    inputValues[289] = 0.054902;
    inputValues[290] = 0.00392157;
    inputValues[291] = 0.603922;
    inputValues[292] = 0.992157;
    inputValues[293] = 0.352941;
    inputValues[294] = 0;
    inputValues[295] = 0;
    inputValues[296] = 0;
    inputValues[297] = 0;
    inputValues[298] = 0;
    inputValues[299] = 0;
    inputValues[300] = 0;
    inputValues[301] = 0;
    inputValues[302] = 0;
    inputValues[303] = 0;
    inputValues[304] = 0;
    inputValues[305] = 0;
    inputValues[306] = 0;
    inputValues[307] = 0;
    inputValues[308] = 0;
    inputValues[309] = 0;
    inputValues[310] = 0;
    inputValues[311] = 0;
    inputValues[312] = 0;
    inputValues[313] = 0;
    inputValues[314] = 0;
    inputValues[315] = 0;
    inputValues[316] = 0;
    inputValues[317] = 0;
    inputValues[318] = 0;
    inputValues[319] = 0.545098;
    inputValues[320] = 0.992157;
    inputValues[321] = 0.745098;
    inputValues[322] = 0.00784314;
    inputValues[323] = 0;
    inputValues[324] = 0;
    inputValues[325] = 0;
    inputValues[326] = 0;
    inputValues[327] = 0;
    inputValues[328] = 0;
    inputValues[329] = 0;
    inputValues[330] = 0;
    inputValues[331] = 0;
    inputValues[332] = 0;
    inputValues[333] = 0;
    inputValues[334] = 0;
    inputValues[335] = 0;
    inputValues[336] = 0;
    inputValues[337] = 0;
    inputValues[338] = 0;
    inputValues[339] = 0;
    inputValues[340] = 0;
    inputValues[341] = 0;
    inputValues[342] = 0;
    inputValues[343] = 0;
    inputValues[344] = 0;
    inputValues[345] = 0;
    inputValues[346] = 0;
    inputValues[347] = 0.0431373;
    inputValues[348] = 0.745098;
    inputValues[349] = 0.992157;
    inputValues[350] = 0.27451;
    inputValues[351] = 0;
    inputValues[352] = 0;
    inputValues[353] = 0;
    inputValues[354] = 0;
    inputValues[355] = 0;
    inputValues[356] = 0;
    inputValues[357] = 0;
    inputValues[358] = 0;
    inputValues[359] = 0;
    inputValues[360] = 0;
    inputValues[361] = 0;
    inputValues[362] = 0;
    inputValues[363] = 0;
    inputValues[364] = 0;
    inputValues[365] = 0;
    inputValues[366] = 0;
    inputValues[367] = 0;
    inputValues[368] = 0;
    inputValues[369] = 0;
    inputValues[370] = 0;
    inputValues[371] = 0;
    inputValues[372] = 0;
    inputValues[373] = 0;
    inputValues[374] = 0;
    inputValues[375] = 0;
    inputValues[376] = 0.137255;
    inputValues[377] = 0.945098;
    inputValues[378] = 0.882353;
    inputValues[379] = 0.627451;
    inputValues[380] = 0.423529;
    inputValues[381] = 0.00392157;
    inputValues[382] = 0;
    inputValues[383] = 0;
    inputValues[384] = 0;
    inputValues[385] = 0;
    inputValues[386] = 0;
    inputValues[387] = 0;
    inputValues[388] = 0;
    inputValues[389] = 0;
    inputValues[390] = 0;
    inputValues[391] = 0;
    inputValues[392] = 0;
    inputValues[393] = 0;
    inputValues[394] = 0;
    inputValues[395] = 0;
    inputValues[396] = 0;
    inputValues[397] = 0;
    inputValues[398] = 0;
    inputValues[399] = 0;
    inputValues[400] = 0;
    inputValues[401] = 0;
    inputValues[402] = 0;
    inputValues[403] = 0;
    inputValues[404] = 0;
    inputValues[405] = 0.317647;
    inputValues[406] = 0.941176;
    inputValues[407] = 0.992157;
    inputValues[408] = 0.992157;
    inputValues[409] = 0.466667;
    inputValues[410] = 0.0980392;
    inputValues[411] = 0;
    inputValues[412] = 0;
    inputValues[413] = 0;
    inputValues[414] = 0;
    inputValues[415] = 0;
    inputValues[416] = 0;
    inputValues[417] = 0;
    inputValues[418] = 0;
    inputValues[419] = 0;
    inputValues[420] = 0;
    inputValues[421] = 0;
    inputValues[422] = 0;
    inputValues[423] = 0;
    inputValues[424] = 0;
    inputValues[425] = 0;
    inputValues[426] = 0;
    inputValues[427] = 0;
    inputValues[428] = 0;
    inputValues[429] = 0;
    inputValues[430] = 0;
    inputValues[431] = 0;
    inputValues[432] = 0;
    inputValues[433] = 0;
    inputValues[434] = 0.176471;
    inputValues[435] = 0.729412;
    inputValues[436] = 0.992157;
    inputValues[437] = 0.992157;
    inputValues[438] = 0.588235;
    inputValues[439] = 0.105882;
    inputValues[440] = 0;
    inputValues[441] = 0;
    inputValues[442] = 0;
    inputValues[443] = 0;
    inputValues[444] = 0;
    inputValues[445] = 0;
    inputValues[446] = 0;
    inputValues[447] = 0;
    inputValues[448] = 0;
    inputValues[449] = 0;
    inputValues[450] = 0;
    inputValues[451] = 0;
    inputValues[452] = 0;
    inputValues[453] = 0;
    inputValues[454] = 0;
    inputValues[455] = 0;
    inputValues[456] = 0;
    inputValues[457] = 0;
    inputValues[458] = 0;
    inputValues[459] = 0;
    inputValues[460] = 0;
    inputValues[461] = 0;
    inputValues[462] = 0;
    inputValues[463] = 0.0627451;
    inputValues[464] = 0.364706;
    inputValues[465] = 0.988235;
    inputValues[466] = 0.992157;
    inputValues[467] = 0.733333;
    inputValues[468] = 0;
    inputValues[469] = 0;
    inputValues[470] = 0;
    inputValues[471] = 0;
    inputValues[472] = 0;
    inputValues[473] = 0;
    inputValues[474] = 0;
    inputValues[475] = 0;
    inputValues[476] = 0;
    inputValues[477] = 0;
    inputValues[478] = 0;
    inputValues[479] = 0;
    inputValues[480] = 0;
    inputValues[481] = 0;
    inputValues[482] = 0;
    inputValues[483] = 0;
    inputValues[484] = 0;
    inputValues[485] = 0;
    inputValues[486] = 0;
    inputValues[487] = 0;
    inputValues[488] = 0;
    inputValues[489] = 0;
    inputValues[490] = 0;
    inputValues[491] = 0;
    inputValues[492] = 0;
    inputValues[493] = 0.976471;
    inputValues[494] = 0.992157;
    inputValues[495] = 0.976471;
    inputValues[496] = 0.25098;
    inputValues[497] = 0;
    inputValues[498] = 0;
    inputValues[499] = 0;
    inputValues[500] = 0;
    inputValues[501] = 0;
    inputValues[502] = 0;
    inputValues[503] = 0;
    inputValues[504] = 0;
    inputValues[505] = 0;
    inputValues[506] = 0;
    inputValues[507] = 0;
    inputValues[508] = 0;
    inputValues[509] = 0;
    inputValues[510] = 0;
    inputValues[511] = 0;
    inputValues[512] = 0;
    inputValues[513] = 0;
    inputValues[514] = 0;
    inputValues[515] = 0;
    inputValues[516] = 0;
    inputValues[517] = 0;
    inputValues[518] = 0.180392;
    inputValues[519] = 0.509804;
    inputValues[520] = 0.717647;
    inputValues[521] = 0.992157;
    inputValues[522] = 0.992157;
    inputValues[523] = 0.811765;
    inputValues[524] = 0.00784314;
    inputValues[525] = 0;
    inputValues[526] = 0;
    inputValues[527] = 0;
    inputValues[528] = 0;
    inputValues[529] = 0;
    inputValues[530] = 0;
    inputValues[531] = 0;
    inputValues[532] = 0;
    inputValues[533] = 0;
    inputValues[534] = 0;
    inputValues[535] = 0;
    inputValues[536] = 0;
    inputValues[537] = 0;
    inputValues[538] = 0;
    inputValues[539] = 0;
    inputValues[540] = 0;
    inputValues[541] = 0;
    inputValues[542] = 0;
    inputValues[543] = 0;
    inputValues[544] = 0.152941;
    inputValues[545] = 0.580392;
    inputValues[546] = 0.898039;
    inputValues[547] = 0.992157;
    inputValues[548] = 0.992157;
    inputValues[549] = 0.992157;
    inputValues[550] = 0.980392;
    inputValues[551] = 0.713726;
    inputValues[552] = 0;
    inputValues[553] = 0;
    inputValues[554] = 0;
    inputValues[555] = 0;
    inputValues[556] = 0;
    inputValues[557] = 0;
    inputValues[558] = 0;
    inputValues[559] = 0;
    inputValues[560] = 0;
    inputValues[561] = 0;
    inputValues[562] = 0;
    inputValues[563] = 0;
    inputValues[564] = 0;
    inputValues[565] = 0;
    inputValues[566] = 0;
    inputValues[567] = 0;
    inputValues[568] = 0;
    inputValues[569] = 0;
    inputValues[570] = 0.0941176;
    inputValues[571] = 0.447059;
    inputValues[572] = 0.866667;
    inputValues[573] = 0.992157;
    inputValues[574] = 0.992157;
    inputValues[575] = 0.992157;
    inputValues[576] = 0.992157;
    inputValues[577] = 0.788235;
    inputValues[578] = 0.305882;
    inputValues[579] = 0;
    inputValues[580] = 0;
    inputValues[581] = 0;
    inputValues[582] = 0;
    inputValues[583] = 0;
    inputValues[584] = 0;
    inputValues[585] = 0;
    inputValues[586] = 0;
    inputValues[587] = 0;
    inputValues[588] = 0;
    inputValues[589] = 0;
    inputValues[590] = 0;
    inputValues[591] = 0;
    inputValues[592] = 0;
    inputValues[593] = 0;
    inputValues[594] = 0;
    inputValues[595] = 0;
    inputValues[596] = 0.0901961;
    inputValues[597] = 0.258824;
    inputValues[598] = 0.835294;
    inputValues[599] = 0.992157;
    inputValues[600] = 0.992157;
    inputValues[601] = 0.992157;
    inputValues[602] = 0.992157;
    inputValues[603] = 0.776471;
    inputValues[604] = 0.317647;
    inputValues[605] = 0.00784314;
    inputValues[606] = 0;
    inputValues[607] = 0;
    inputValues[608] = 0;
    inputValues[609] = 0;
    inputValues[610] = 0;
    inputValues[611] = 0;
    inputValues[612] = 0;
    inputValues[613] = 0;
    inputValues[614] = 0;
    inputValues[615] = 0;
    inputValues[616] = 0;
    inputValues[617] = 0;
    inputValues[618] = 0;
    inputValues[619] = 0;
    inputValues[620] = 0;
    inputValues[621] = 0;
    inputValues[622] = 0.0705882;
    inputValues[623] = 0.670588;
    inputValues[624] = 0.858824;
    inputValues[625] = 0.992157;
    inputValues[626] = 0.992157;
    inputValues[627] = 0.992157;
    inputValues[628] = 0.992157;
    inputValues[629] = 0.764706;
    inputValues[630] = 0.313726;
    inputValues[631] = 0.0352941;
    inputValues[632] = 0;
    inputValues[633] = 0;
    inputValues[634] = 0;
    inputValues[635] = 0;
    inputValues[636] = 0;
    inputValues[637] = 0;
    inputValues[638] = 0;
    inputValues[639] = 0;
    inputValues[640] = 0;
    inputValues[641] = 0;
    inputValues[642] = 0;
    inputValues[643] = 0;
    inputValues[644] = 0;
    inputValues[645] = 0;
    inputValues[646] = 0;
    inputValues[647] = 0;
    inputValues[648] = 0.215686;
    inputValues[649] = 0.67451;
    inputValues[650] = 0.886275;
    inputValues[651] = 0.992157;
    inputValues[652] = 0.992157;
    inputValues[653] = 0.992157;
    inputValues[654] = 0.992157;
    inputValues[655] = 0.956863;
    inputValues[656] = 0.521569;
    inputValues[657] = 0.0431373;
    inputValues[658] = 0;
    inputValues[659] = 0;
    inputValues[660] = 0;
    inputValues[661] = 0;
    inputValues[662] = 0;
    inputValues[663] = 0;
    inputValues[664] = 0;
    inputValues[665] = 0;
    inputValues[666] = 0;
    inputValues[667] = 0;
    inputValues[668] = 0;
    inputValues[669] = 0;
    inputValues[670] = 0;
    inputValues[671] = 0;
    inputValues[672] = 0;
    inputValues[673] = 0;
    inputValues[674] = 0;
    inputValues[675] = 0;
    inputValues[676] = 0.533333;
    inputValues[677] = 0.992157;
    inputValues[678] = 0.992157;
    inputValues[679] = 0.992157;
    inputValues[680] = 0.831373;
    inputValues[681] = 0.529412;
    inputValues[682] = 0.517647;
    inputValues[683] = 0.0627451;
    inputValues[684] = 0;
    inputValues[685] = 0;
    inputValues[686] = 0;
    inputValues[687] = 0;
    inputValues[688] = 0;
    inputValues[689] = 0;
    inputValues[690] = 0;
    inputValues[691] = 0;
    inputValues[692] = 0;
    inputValues[693] = 0;
    inputValues[694] = 0;
    inputValues[695] = 0;
    inputValues[696] = 0;
    inputValues[697] = 0;
    inputValues[698] = 0;
    inputValues[699] = 0;
    inputValues[700] = 0;
    inputValues[701] = 0;
    inputValues[702] = 0;
    inputValues[703] = 0;
    inputValues[704] = 0;
    inputValues[705] = 0;
    inputValues[706] = 0;
    inputValues[707] = 0;
    inputValues[708] = 0;
    inputValues[709] = 0;
    inputValues[710] = 0;
    inputValues[711] = 0;
    inputValues[712] = 0;
    inputValues[713] = 0;
    inputValues[714] = 0;
    inputValues[715] = 0;
    inputValues[716] = 0;
    inputValues[717] = 0;
    inputValues[718] = 0;
    inputValues[719] = 0;
    inputValues[720] = 0;
    inputValues[721] = 0;
    inputValues[722] = 0;
    inputValues[723] = 0;
    inputValues[724] = 0;
    inputValues[725] = 0;
    inputValues[726] = 0;
    inputValues[727] = 0;
    inputValues[728] = 0;
    inputValues[729] = 0;
    inputValues[730] = 0;
    inputValues[731] = 0;
    inputValues[732] = 0;
    inputValues[733] = 0;
    inputValues[734] = 0;
    inputValues[735] = 0;
    inputValues[736] = 0;
    inputValues[737] = 0;
    inputValues[738] = 0;
    inputValues[739] = 0;
    inputValues[740] = 0;
    inputValues[741] = 0;
    inputValues[742] = 0;
    inputValues[743] = 0;
    inputValues[744] = 0;
    inputValues[745] = 0;
    inputValues[746] = 0;
    inputValues[747] = 0;
    inputValues[748] = 0;
    inputValues[749] = 0;
    inputValues[750] = 0;
    inputValues[751] = 0;
    inputValues[752] = 0;
    inputValues[753] = 0;
    inputValues[754] = 0;
    inputValues[755] = 0;
    inputValues[756] = 0;
    inputValues[757] = 0;
    inputValues[758] = 0;
    inputValues[759] = 0;
    inputValues[760] = 0;
    inputValues[761] = 0;
    inputValues[762] = 0;
    inputValues[763] = 0;
    inputValues[764] = 0;
    inputValues[765] = 0;
    inputValues[766] = 0;
    inputValues[767] = 0;
    inputValues[768] = 0;
    inputValues[769] = 0;
    inputValues[770] = 0;
    inputValues[771] = 0;
    inputValues[772] = 0;
    inputValues[773] = 0;
    inputValues[774] = 0;
    inputValues[775] = 0;
    inputValues[776] = 0;
    inputValues[777] = 0;
    inputValues[778] = 0;
    inputValues[779] = 0;
    inputValues[780] = 0;
    inputValues[781] = 0;
    inputValues[782] = 0;
    inputValues[783] = 0;
}
