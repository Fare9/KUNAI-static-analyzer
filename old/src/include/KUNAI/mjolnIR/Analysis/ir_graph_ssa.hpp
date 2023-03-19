/**
 * @file ir_graph_ssa.hpp
 * @author Farenain
 * @brief This is an extension of a IRGraph class, but in this case
 *        the nodes will contain a SSA form of the graph, where each
 *        variable will be assigned only once, as parameter the constructor
 *        will receive an IRGraph and this will be transformed to the
 *        IRGraphSSA.
 */

#include <optional>
#include <unordered_map>
#include <stack>

#include "KUNAI/mjolnIR/ir_graph.hpp"

namespace KUNAI
{
    namespace MJOLNIR
    {
        class IRGraphSSA;

        using irgraphssa_t = std::shared_ptr<IRGraphSSA>;

        class IRGraphSSA : public IRGraph
        {
        public:
            /**
             * @brief Construct a new IRGraphSSA object for this it will
             *        be necessary a code IRGraph.
             *
             * @param code_graph graph used for constructing the SSA Graph.
             */
            IRGraphSSA(irgraph_t &code_graph);

            /**
             * @brief Destroy the IRGraphSSA object
             */
            ~IRGraphSSA() = default;

        private:
            std::unordered_map<irreg_t, std::set<irblock_t>> var_block_map;

            std::unordered_map<irreg_t, std::uint32_t> C;
            std::unordered_map<irreg_t, std::stack<irreg_t>> S;
            std::unordered_map<irreg_t, irreg_t> ssa_to_non_ssa_form;

            std::map<KUNAI::MJOLNIR::irblock_t, KUNAI::MJOLNIR::irblock_t> dominance_tree;

            /**
             * @brief Obtain all kind of assignment that can generate
             *        a newer value for a variable.
             *
             * @param graph
             */
            void collect_var_assign();

            /**
             * @brief Look for a place in the dominance frontier where
             *        to write
             *
             */
            void insert_phi_node();

            /**
             * @brief Apply variable renaming to a basic block of the IRGraph
             *        here we will apply the global variables in order to translate
             *        each instruction.
             *
             * @param v basic block to translate.
             */
            void search(const irblock_t &v);

            /**
             * @brief Translate an instruction to an SSA form this will involve
             *        parsing the instruction and checking if it contains registers
             *        to translate to a new SSA form.
             *
             * @param instr instruction to translate to an SSA form
             * @param p defined registers that we must remove later from stack.
             * @return irstmnt_t
             */
            irstmnt_t translate_instruction(irstmnt_t &instr, std::list<irreg_t> &p);

            /**
             * @brief Create a new register for the SSA, this will be used
             *        in the renaming algorithm, the algorithm is based in
             *        the one of the book
             *        "An Introduction to the Theory of Optimizing Compilers".
             *        This function will use both C and S.
             *
             * @param old_reg register we want to transform to SSA
             * @param p defined registers that we must remove later from stack.
             * @return irreg_t
             */
            irreg_t create_new_ssa_reg(irreg_t old_reg, std::list<irreg_t> &p);

            /**
             * @brief Get the top of the S stack for a given register, in case
             *        it doesn't exist yet, call to create_new_ssa_reg function.
             * 
             * @param old_reg register we want to transform to SSA
             * @param p definedd registers that we must remove later from stack
             * @return irreg_t 
             */
            irreg_t get_top_or_create(irreg_t old_reg, std::list<irreg_t> &p);
        };
    } //! MJOLNIR
} //! KUNAI