/***
 * File: dex.hpp
 * Author: @Farenain
 * 
 * Class which represents a dex file.
 */

#pragma once

#ifndef DEX_HPP
#define HEX_HPP

#include <iostream>
#include <memory>

#include "dex_parser.hpp"

namespace KUNAI {
    namespace DEX {
        class DEX {
        public:
            DEX(std::ifstream& input_file, std::uint64_t file_size);
            ~DEX();

            std::shared_ptr<DexParser> get_parser();
            
        private:
            std::shared_ptr<DexParser> dex_parser;
            bool dex_parsing_correct;
        };
    }
}


#endif