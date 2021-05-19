/***
 * @file dex.hpp
 * @author @Farenain
 * 
 * @brief Class to manage a DEX file.
 * 
 * This class must serve to the user as an starting
 * point of the analysis of a DEX file, here user should
 * have access to analysis objets for classes, methods,
 * instructions and so on.
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