//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file kunaistream.hpp
// @brief Manage the possible streams for reading and writing, as well as
// possibly reading data types that need special handling.
#ifndef KUNAI_UTILS_KUNAISTREAM_HPP
#define KUNAI_UTILS_KUNAISTREAM_HPP

#include <fstream>
#include "Kunai/Exceptions/stream_exception.hpp"

namespace KUNAI
{
    namespace stream
    {
        /// @brief Class to manage an input file stream given for the analysis.
        class KunaiStream
        {
            /// @brief Input file to read from
            std::ifstream& input_file;
            
            /// @brief Size of the file
            std::size_t file_size;

            /// @brief any initialization from the stream do it here
            void initialize();

        public:
            /// @brief maximum size for ansii strings
            const std::int32_t MAX_ANSII_STR_SIZE = 256;

            /// @brief Constructor from KunaiStream class.
            /// @param input_file file for the analysis this must be an std::ifstream
            KunaiStream(std::ifstream& input_file) : input_file(input_file)
            {
                if (!input_file.is_open())
                    throw exceptions::StreamException("KunaiStream: error input_file not open");
                initialize();
            }

            /// @brief Destructor from KunaiStream, nothing done here
            /// for the moment
            ~KunaiStream() = default;

            /// @brief Obtain the size of the file
            /// @return std::size_t with the calculated file size
            std::size_t get_size() const
            {
                return file_size;
            }

            /// @brief Read data given a buffer of a T data type and with
            /// a size specified by the user
            /// @tparam T type of the buffer where to read the data
            /// @param buffer Buffer where to read the data from the file.
            /// @param read_size size to read
            template <typename T>
            void read_data(T& buffer, std::int32_t read_size)
            {
                if (read_size < 0)
                    throw exceptions::StreamException("read_data(): read_size given incorrect");
                // read the data
                input_file.read(reinterpret_cast<char*>(&buffer), read_size);

                if(!input_file)
                    throw exceptions::StreamException("read_data(): error reading input file");
            }

            /// @brief Move the pointer from the input file
            /// @param off offset where to move
            /// @param dir directorion to move
            void seekg(std::streamoff off, std::ios_base::seekdir dir)
            {
                input_file.seekg(off, dir);
            }

            /// @brief Read a string as an array of char finished in a 0 byte
            /// @param offset the offset in the file where to read the string
            /// @return string read
            std::string read_ansii_string(std::int64_t offset);

            /// @brief Read a DEX string, the dex string contains the next format:
            /// <size in uleb128><string with size>
            /// @param offset the offset in the file where to read the string
            /// @return string read
            std::string read_dex_string(std::int64_t offset);

            /// @brief Read a number in uleb128 format.
            /// @return uint64_t with the number
            std::uint64_t read_uleb128();

            /// @brief Read a number in sleb128 format.
            /// @return int64_t with the number
            std::int64_t read_sleb128();
        };
    }   
} // namespace KUNAI


#endif // KUNAI_UTILS_KUNAISTREAM_HPP