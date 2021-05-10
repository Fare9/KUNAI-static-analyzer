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
#include "dex_header.hpp"
#include "dex_strings.hpp"
#include "dex_types.hpp"
#include "dex_protos.hpp"
#include "dex_fields.hpp"
#include "dex_methods.hpp"
#include "dex_classes.hpp"

namespace KUNAI {
    namespace DEX {
        class DEX {
        public:
            DEX(std::ifstream& input_file, std::uint64_t file_size);
            ~DEX();

            friend std::ostream& operator<<(std::ostream& os, const DEX& entry);
        private:
            std::shared_ptr<DexHeader> dex_header       = nullptr;
            std::shared_ptr<DexStrings> dex_strings     = nullptr;
            std::shared_ptr<DexTypes> dex_types         = nullptr;
            std::shared_ptr<DexProtos> dex_protos       = nullptr;
            std::shared_ptr<DexFields> dex_fields       = nullptr;
            std::shared_ptr<DexMethods> dex_methods     = nullptr;
            std::shared_ptr<DexClasses> dex_classes     = nullptr;
        };
    }
}


#endif