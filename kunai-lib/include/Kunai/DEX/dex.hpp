//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex.hpp
// @brief Managing of DEX files, the DEX files are managed by a generic class
// this will check the DEX file is correct, will parse and optionally will
// analyze the DEX.
#ifndef KUNAI_DEX_DEX_HPP
#define KUNAI_DEX_DEX_HPP

//! utilities
#include "Kunai/Utils/logger.hpp"
#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/DEX/parser/parser.hpp"
#include "Kunai/DEX/DVM/dex_disassembler.hpp"
#include "Kunai/DEX/analysis/dex_analysis.hpp"

#include <memory>

namespace KUNAI
{
namespace DEX
{
    /// @brief Abstraction of a DEX object, the class offers the analyst
    /// a parser, a disassembler and an analysis object.
    class Dex
    {
    public:
        /// @brief Parse a given dex file, return a Dex object as a unique pointer
        /// @param dex_file_path path to a dex file
        /// @return unique pointer with Dex object
        static std::unique_ptr<Dex> parse_dex_file(std::string& dex_file_path);

        static std::unique_ptr<Dex> parse_dex_file(char * dex_file_path);

        static std::unique_ptr<Dex> parse_dex_file(const char * dex_file_path);
    
    private:
        /// @brief Stream to manage the DEX file with utilities
        std::unique_ptr<stream::KunaiStream> kunai_stream;
        /// @brief ifstream that will hold the dex file
        std::ifstream dex_file;

        /// @brief Parser for the DEX file
        std::unique_ptr<Parser> parser;

        /// @brief Disassembler for DEX file
        std::unique_ptr<DexDisassembler> dex_disassembler;

        /// @brief Analysis object for DEX file
        std::unique_ptr<Analysis> analysis;

        /// @brief store if parsing process was correct
        bool parsing_correct = false;

        /// @brief Method to initialize structures from the DEX
        /// or open files, etc
        /// @param dex_file_path path to a dex file
        void initialization(std::string& dex_file_path);

    public:

        /// @brief Constructor of the Dex object, we obtain a path
        /// to the DEX file to analyze.
        /// @param dex_file_path path to a dex file
        Dex(std::string& dex_file_path)
        {
            initialization(dex_file_path);
        }

        /// @brief Destructor of the Dex object, release any memory
        /// or files here in case it is needed.
        ~Dex()
        {
            if (dex_file.is_open())
                dex_file.close();
        }

        /// @brief Was parsing process correct?
        /// @return boolean with result
        bool get_parsing_correct() const
        {
            return parsing_correct;
        }

        /// @brief get a pointer to the DEX parser with all the headers
        /// @return dex parser
        Parser * get_parser()
        {
            return parser.get();
        }

        /// @brief Get the disassembler of the DEX file
        /// @return pointer to DexDisassembler
        DexDisassembler * get_dex_disassembler()
        {
            return dex_disassembler.get();
        }
        
        /// @brief Get the analysis object this needs the
        /// disassembly and the parser, the dex will be
        /// disassembled in case this function is called
        /// @param create_xrefs create all the xrefs for the
        /// class, this can take a long time
        /// @return pointer to Analysis object or nullptr
        /// in case of error
        Analysis * get_analysis(bool create_xrefs);
    };

} // namespace DEX
} // namespace KUNAI


#endif // KUNAI_DEX_DEX_HPP