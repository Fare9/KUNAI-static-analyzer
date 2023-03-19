/**
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

#ifndef DEX_STRINGS_HPP
#define DEX_STRINGS_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <iterator>

#include "KUNAI/Exceptions/exceptions.hpp"
#include "KUNAI/Utils/utils.hpp"

namespace KUNAI {
    namespace DEX {

        class DexStrings;

        using dexstrings_t = std::unique_ptr<DexStrings>;
        
        using offset_string_t = std::map<std::uint32_t, std::unique_ptr<std::string>>;

        using ordered_strings_t = std::vector<std::string*>;


        class DexStrings
        {
        public:
            /**
             * @brief Generate new DexStrings object.
             * @param input_file: file where to read DexString.
             * @param file_size: size of file for checks.
             * @param number_of_strings: number of different strings from DEX.
             * @param strings_offsets: offset where to read the strings.
             */
            DexStrings(std::ifstream& input_file, 
                        std::uint64_t file_size, 
                        std::uint32_t number_of_strings, 
                        std::uint32_t strings_offsets);

            /**
             * @brief DexStrings destructor, clear strings vector.
             */
            ~DexStrings();

            /**
             * @brief Return a string pointer by the offset.
             * @param offset: string offset in DEX file.
             * @return std::string* string from DEX file.
             */
            std::string* get_string_from_offset(std::uint32_t offset);
            
            /**
             * @brief Return a string pointer by order of the string.
             * @param pos: position where toget the string.
             * @return std::string* string from DEX file.
             */
            std::string* get_string_from_order(std::uint32_t pos);
            
            /**
             * @brief Return all the strings in a vector.
             * @return const std::vector<std::string*> const: all the list of strings.
             */
            const ordered_strings_t &get_all_strings() const
            {
                return ordered_strings;
            }

            /**
             * @brief Get number of all the DEX strings.
             * @return number of all DEX strings.
             */
            std::uint32_t get_number_of_strings()
            {
                return number_of_strings;
            }

            /**
             * @brief pretty print strings
             * @return std::ostream with strings pretty printed
             */
            friend std::ostream& operator<<(std::ostream& os, const DexStrings& entry);
            
            /**
             * @brief dump to a std::fstream the strings in XML format.
             * @return std::fstream with strings in XML format
             */
            friend std::fstream& operator<<(std::fstream& fos, const DexStrings& entry);
        private:
            // private methods

            /**
             * @brief private method to parse strings of DEX file.
             * @param input_file: file where to search the strings.
             * @param file_size: size of file for checks.
             * @return true if correct, false if a problem happen.
             */
            bool parse_strings(std::ifstream& input_file, std::uint64_t file_size);

            // variables from strings
            std::uint32_t number_of_strings;
            std::uint32_t offset;
            offset_string_t strings;
            ordered_strings_t ordered_strings;
        };

    }
}


#endif
