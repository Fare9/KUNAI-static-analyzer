//--------------------------------------------------------------------*- C++ -*-
// backward-slicing
// @author sunniAngela
//
// @file backward_analysis.hpp
// @brief Definition of a BackwardAnalysis class that contains methods to
// perform a simple backward slicing using the Analysis module of Kunai

#include <iostream>
#include <fstream>
#include <regex>

#include <Kunai/DEX/dex.hpp>

namespace KUNAI
{
namespace DEX
{
    class BackwardAnalysis{
        public:
            struct target_info {
                std::string class_name;
                std::string method_name;
                std::string prototype;
                bool is_parsed_correct;
            };

            // Vector containing the parsed information of each target method
            std::vector<target_info> targets_info;

            // Vector containing the register number of each of the parameters of a target method
            std::vector<std::uint8_t> parameters_registers;

            // Map for register number and the vector of its defining instructions
            std::unordered_map<std::uint8_t, std::vector<KUNAI::DEX::Instruction*>> instruction_mapped_registers;

            // Map for register number and its defining instruction (first occurence)
            std::unordered_map<std::uint8_t, KUNAI::DEX::Instruction*> instruction_mapped_registers_single;

            /// @brief Parse the targets file and extracts the corresponding data from each line
            /// @param file_path path of the file to extract information
            /// @return error code
            int8_t parse_targets_file(std::string file_path);

            /// @brief Find the invoking instruction in the xreffrom and the registers corresponding to the parameters
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            /// @param target_method MethodAnalysis object for the target method
            /// @param addr address of the invoke instruction
            void find_parameters_registers(KUNAI::DEX::MethodAnalysis* xreffrom_method, KUNAI::DEX::MethodAnalysis* target_method, std::uint64_t addr);

            /// @brief Read instructions from the basic block in reverse to find the instruction that defines the
            /// given register
            /// @param bb Basic block
            /// @param reg Register value
            /// @return 1 if the defining instruction has been found in the block, 0 otherwise
            int8_t analyze_block(KUNAI::DEX::DVMBasicBlock* bb, uint8_t reg);

            /// @brief Find the invoking instruction in the xreffrom and the registers corresponding to a specific
            /// parameter given by its register
            /// @param instruc Instruction to analyze
            /// @param pr Value of the register holding the parameter
            /// @return 1 if the instruction defines the register, 0 otherwise
            int8_t find_defining_instruction(KUNAI::DEX::Instruction* instruc, uint8_t pr);

            /// @brief Find the invoking instruction in the xreffrom and the registers corresponding to the parameters
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            /// @param target_method MethodAnalysis object for the target method
            void find_defining_instruction_single(KUNAI::DEX::Instruction* instruc);
            
            /// @brief Find the instruction that defines each parameter of the target method from its 'xreffrom' 
            /// by basic blocks
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            void find_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method);

            /// @brief Find the instruction that defines each parameter of the target method from its 'xreffrom' 
            /// by instructions (first occurrence)
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            void find_parameters_definition_single(KUNAI::DEX::MethodAnalysis *xreffrom_method);

            /// @brief Print the instruction where the parameters are first defined or the method name if it was
            /// defined as a method parameter
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            void print_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method);

            /// @brief Print the instruction where the parameters are first defined or the method name if it was
            /// defined as a method parameter
            /// @param xreffrom_method MethodAnalysis object for the xref (caller) method
            void print_parameters_definition_single(KUNAI::DEX::MethodAnalysis *xreffrom_method);
    };

} // namespace DEX
} // namespace KUNAI