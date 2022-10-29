#include "KUNAI/DEX/parser/dex_strings.hpp"

namespace KUNAI
{
    namespace DEX
    {

        DexStrings::DexStrings(std::ifstream &input_file,
                               std::uint64_t file_size,
                               std::uint32_t number_of_strings,
                               std::uint32_t strings_offsets) : number_of_strings(number_of_strings),
                                                                offset(strings_offsets)
        {
            if (!parse_strings(input_file, file_size))
                throw exceptions::ParserReadingException("Error reading DEX strings");
        }

        DexStrings::~DexStrings() = default;

        std::string *DexStrings::get_string_from_offset(std::uint32_t offset)
        {
            if (strings.find(offset) == strings.end())
                return nullptr;

            return (strings[offset].get());
        }

        std::string *DexStrings::get_string_from_order(std::uint32_t pos)
        {
            if (pos >= strings.size())
                return nullptr;

            return ordered_strings[pos];
        }

        /**
         * Private methods
         */

        bool DexStrings::parse_strings(std::ifstream &input_file, std::uint64_t file_size)
        {
            auto logger = LOGGER::logger();

            auto current_offset = input_file.tellg();
            size_t i;
            // string values
            std::uint32_t str_offset;
            std::string str;

            // move to offset where are the string ids
            input_file.seekg(offset);

            #ifdef DEBUG
            logger->debug("DexStrings parsing of header in offset {} with size {}", offset, number_of_strings);
            #endif

            // go one by one reading offset and string
            for (i = 0; i < number_of_strings; i++)
            {
                if (!KUNAI::read_data_file<std::uint32_t>(str_offset, sizeof(std::uint32_t), input_file))
                    return false;

                if (str_offset > file_size)
                {
                    logger->error("Error offset from string out of file bound ({} > {})", str_offset, file_size);
                    throw exceptions::OutOfBoundException("Error offset from string out of file bound");
                }

                str = KUNAI::read_dex_string(input_file, str_offset);

                std::unique_ptr<std::string> p_str = std::make_unique<std::string>(str);
                
                strings[str_offset] = std::move(p_str);
                ordered_strings.push_back(p_str.get());

                #ifdef DEBUG
                logger->debug("parsed string number {}", i);
                #endif
            }

            input_file.seekg(current_offset);

            logger->info("DexStrings parsing correct");

            return true;
        }

        /**
         * friend methods
         */

        std::ostream &operator<<(std::ostream &os, const DexStrings &entry)
        {
            size_t i = 0;
            os << std::hex;
            os << std::setw(30) << std::left << std::setfill(' ') << "=========== DEX Strings ===========" << "\n";
            for (const auto& s : entry.strings)
                os << std::left << std::setfill(' ') << "String (" << std::dec << i++ << std::hex << "): " << s.first << "->\"" << s.second << "\"" << "\n";

            return os;
        }

        std::fstream &operator<<(std::fstream &fos, const DexStrings &entry)
        {
            std::stringstream stream;

            stream << std::hex;
            stream << "<strings>" << "\n";
            for (const auto& s : entry.strings)
            {
                stream << "\t<string>" << "\n";
                stream << "\t\t<offset>" << s.first << "</offset>" << "\n";
                stream << "\t\t<value>" << *s.second << "</value>" << "\n";
                stream << "\t</string>" << "\n";
            }
            
            stream << "</strings>" << "\n";

            fos.write(stream.str().c_str(), stream.str().size());

            return fos;
        }
    }
}