//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file dex_disassembler.hpp
// @brief This class holds the different disassembly algorithms for DEX
// the user can retrieve any of them and use if for disassembly of the
// methods, internally dex_disassembler will provide the algorithms with
// an object of Disassembler class

#ifndef KUNAI_DEX_DVM_DEX_DISASSEMBLER_HPP
#define KUNAI_DEX_DVM_DEX_DISASSEMBLER_HPP

#include "Kunai/DEX/parser/methods.hpp"
#include "Kunai/DEX/parser/classes.hpp"
#include "Kunai/DEX/DVM/disassembler.hpp"
#include "Kunai/DEX/DVM/linear_sweep_disassembler.hpp"
#include "Kunai/DEX/DVM/recursive_traversal_disassembler.hpp"

namespace KUNAI
{
namespace DEX
{
    /// @brief Disassembler for DEX data
    class DexDisassembler
    {
    public:
        /// @brief enum of the different algorithms implemented
        /// for doing disassembly of DEX file
        enum class disassembly_algorithm
        {
            LINEAR_SWEEP_ALGORITHM,
            RECURSIVE_TRAVERSAL_ALGORITHM
        };

        
    private:
        /// @brief Algorithm that will be used for
        /// the disassembly process, by default use
        /// the linear sweep
        disassembly_algorithm algorithm = disassembly_algorithm::LINEAR_SWEEP_ALGORITHM;
        /// @brief Disassembler that provides low level
        /// functionalities of a DEX disassembler.
        Disassembler disassembler;

        /// @brief Pointer to the parser in order to apply
        /// disassembly to all dex file
        Parser * parser;

        /// @brief Obtain if disassembly was correct
        bool disassembly_correct = false;

        /// @brief An object containing all the instructions from the
        /// dex file
        Disassembler::instructions_t dex_instructions;

        /// @brief Linear sweep disassembler
        LinearSweepDisassembler linear_sweep;

        /// @brief Recursive Traversal Disassembler
        RecursiveTraversalDisassembler recursive_traversal;
    public:

        /// @brief Constructor of the DexDisassembler, this should be called
        /// only if the parsing was correct
        /// @param parser parser for the internal disassembler, this is used
        /// in some of the instructions
        DexDisassembler(Parser * parser) :
            parser(parser)
        {
            disassembler.set_parser(parser);
            linear_sweep.set_internal_disassembler(&disassembler);
            recursive_traversal.set_internal_disassembler(&disassembler);
        }

        /// @brief Set the disassembly algorithm to use in the next calls to
        /// the different disassembly methods.
        /// @param algorithm new algorithm to use
        void set_disassembly_algorithm(disassembly_algorithm algorithm)
        {
            this->algorithm = algorithm;
        }

        /// @brief Get access to all the instructions from a dex
        /// @return constant reference to instructions
        const Disassembler::instructions_t& get_dex_instructions() const
        {
            return dex_instructions;
        }

        /// @brief Get access to all the instructions from a dex
        /// @return reference to instructions
        Disassembler::instructions_t& get_dex_instructions()
        {
            return dex_instructions;
        }

        /// @brief Get if the disassembly process was correct or not
        /// @return boolean value that says if disassembly was correct
        bool correct_disassembly() const
        {
            return disassembly_correct;
        }

        /// @brief Add the instructions from another disassembler to the
        /// current one, this invalidate the previous one.
        /// @param other other dex disassembler
        void add_disassembly(DexDisassembler& other);

        /// @brief This is the most important function from the
        /// disassembler, this function takes the given parser
        /// object and calls one of the internal disassemblers
        /// for retrieving all the instructions from the DEX file
        void disassembly_dex();

        /// @brief Disassembly a buffer of bytes, take the buffer
        /// of bytes as dalvik instructions
        /// @param buffer buffer with possible bytecode for dalvik
        /// @return vector with disassembled instructions
        std::vector<std::unique_ptr<Instruction>>
            disassembly_buffer(std::vector<std::uint8_t>& buffer);

        DexDisassembler& operator+=(DexDisassembler& other);
    };
} // namespace DEX
} // namespace KUNAI


#endif