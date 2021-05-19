/***
 * @file dex_strings.hpp
 * @author @Farenain
 * 
 * @brief DEX file strings, these are the
 *        offsets in the file, we will store
 *        the strings too.
 * 
 * string_id_item{
 *  uint string_data_off; ---> |
 * }                           |
 *                             |
 * -----------------------------
 * |
 * v 
 * string_data_item{
 *  uleb128 utf16_size;
 *  ubyte[] data;
 * }
 * 
 * string_ids string_id_item[]
 * 
 */

#pragma once

#ifndef DEX_STRINGS_HPP
#define DEX_STRINGS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include "exceptions.hpp"
#include "utils.hpp"

namespace KUNAI {
    namespace DEX {

        class DexStrings
        {
        public:
            DexStrings(std::ifstream& input_file, 
                        std::uint64_t file_size, 
                        std::uint32_t number_of_strings, 
                        std::uint32_t strings_offsets);
            ~DexStrings();

            std::string* get_string_from_offset(std::uint32_t offset);
            std::string* get_string_from_order(std::uint32_t pos);
            std::vector<std::string> get_all_strings();

            std::uint32_t get_number_of_strings();

            friend std::ostream& operator<<(std::ostream& os, const DexStrings& entry);
            friend std::fstream& operator<<(std::fstream& fos, const DexStrings& entry);
        private:
            // private methods
            bool parse_strings(std::ifstream& input_file, std::uint64_t file_size);

            // variables from strings
            std::uint32_t number_of_strings;
            std::uint32_t offset;
            std::map<std::uint32_t, std::string> strings;
            std::vector<std::string*> ordered_strings;
        };

    }
}


#endif