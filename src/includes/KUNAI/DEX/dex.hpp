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
#include "dex_dalvik_opcodes.hpp"
#include "dex_disassembler.hpp"
#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        class DEX
        {
        public:
            DEX(std::ifstream &input_file, std::uint64_t file_size);
            ~DEX();

            std::shared_ptr<DexParser> get_parser();
            std::shared_ptr<DalvikOpcodes> get_dalvik_opcode_object();
            std::shared_ptr<DexDisassembler> get_dex_disassembler();
            std::shared_ptr<Analysis> get_dex_analysis();

            bool get_parsing_correct();
            

        private:
            std::shared_ptr<DexParser> dex_parser;
            std::shared_ptr<DalvikOpcodes> dalvik_opcodes;
            std::shared_ptr<DexDisassembler> dex_disassembler;
            std::shared_ptr<Analysis> dex_analysis;
            

            std::map<std::tuple<std::shared_ptr<ClassDef>,std::shared_ptr<EncodedMethod>>, std::map<std::uint64_t, std::shared_ptr<Instruction>>> method_instructions;

            bool dex_parsing_correct;
        };

        std::unique_ptr<DEX> get_unique_dex_object(std::ifstream &input_file, std::uint64_t file_size);
        std::shared_ptr<DEX> get_shared_dex_object(std::ifstream &input_file, std::uint64_t file_size);
    }
}

#endif