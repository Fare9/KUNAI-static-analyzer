//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file analysis.hpp
// @brief Here we will have all the classes useful for analyst, for example
// accessing to cross-references, and easy accessing data from the DEX file.

#ifndef KUNAI_DEX_ANALYSIS_ANALYSIS_HPP
#define KUNAI_DEX_ANALYSIS_ANALYSIS_HPP

#include "Kunai/DEX/parser/parser.hpp"
#include "Kunai/DEX/DVM/disassembler.hpp"
#include "Kunai/DEX/analysis/external_class.hpp"
#include "Kunai/DEX/DVM/dalvik_instructions.hpp"
#include "Kunai/Exceptions/analysis_exception.hpp"
#include "Kunai/Utils/iterator_range.hpp"

#include <set>
#include <variant>

namespace KUNAI
{
    namespace DEX
    {
        // forward declaration for the xrefs
        class ClassAnalysis;
        class MethodAnalysis;
        class FieldAnalysis;
        class BasicBlocks;

        using instructioniterator_t = std::vector<Instruction *>::iterator;
        using reverseinstructioniterator_t = std::vector<Instruction *>::reverse_iterator;

        using uniqueinstructioniterator_t = std::vector<std::unique_ptr<Instruction>>::iterator;
        using reverseuniqueinstructioniterator_t = std::vector<std::unique_ptr<Instruction>>::reverse_iterator;

        /// @brief Class that contain the instructions of basic block
        /// different DVMBasicBlock exists
        class DVMBasicBlock
        {
        private:
            /// @brief instructions of the current basic block
            std::vector<Instruction *> instructions_;

            /// @brief is a try block?
            bool try_block = false;
            /// @brief is a catch block?
            bool catch_block = false;
            /// @brief is start block? (empty block)
            bool start_block = false;
            /// @brief is end block? (empty block)
            bool end_block = false;
            /// @brief Handler type
            DVMType *handler_type;

            /// @brief name of the block composed by
            /// first and last address
            std::string name;

        private:
            instructioniterator_t instructions_begin() { return instructions_.begin(); }
            instructioniterator_t instructions_end() { return instructions_.end(); }

            reverseinstructioniterator_t instructions_rbegin() { return instructions_.rbegin(); }
            reverseinstructioniterator_t instructions_rend() { return instructions_.rend(); }

        public:
            /// @brief Return a range for going forward through the instructions
            /// @return forward range for instructions
            iterator_range<instructioniterator_t> instructions() { return make_range(instructions_begin(), instructions_end()); }

            /// @brief Return a range for going backward through the instructiosn
            /// @return backward range for instructions
            iterator_range<reverseinstructioniterator_t> reverse_instructions() { return make_range(instructions_rbegin(), instructions_rend()); }

        public:
            DVMBasicBlock() = default;

            /// avoid any kind of copy constructor
            DVMBasicBlock(const DVMBasicBlock &temp_obj) = delete;
            DVMBasicBlock &operator=(const DVMBasicBlock &temp_obj) = delete;

        public:
            /// @brief Obtain the number of instructions from the instructions vector
            /// @return number of instructions of DVMBasicBlock
            size_t get_nb_instructions() const
            {
                return instructions_.size();
            }

            /// @brief Add an instruction to the basic block
            /// @param instr new instructions from the basic block
            void add_instruction(Instruction *instr)
            {
                instructions_.push_back(instr);
            }

            /// @brief Get a constant reference to the vector of instructions
            /// @return instructions from the block
            const std::vector<Instruction *> &get_instructions() const
            {
                return instructions_;
            }

            /// @brief Get a reference to the vector of instructions
            /// @return instructions from the block
            std::vector<Instruction *> &get_instructions()
            {
                return instructions_;
            }

            /// @brief Return the last instruction in case this is a terminator instruction
            /// @return terminator instruction
            Instruction *get_terminator()
            {
                if (instructions_.size() > 0 && instructions_.back()->is_terminator())
                    return instructions_.back();
                return nullptr;
            }

            /// @brief Get the first address of the basic block in case there are instructions
            /// @return first address of basic block
            std::uint64_t get_first_address()
            {
                if (instructions_.size() > 0)
                    return instructions_[0]->get_address();
                throw exceptions::AnalysisException("get_first_address(): basic block has no instructions");
            }

            /// @brief Get the last address of the basic block in case there are instructions
            /// @return last address of basic block
            std::uint64_t get_last_address()
            {
                if (instructions_.size() > 0)
                    return instructions_.back()->get_address();
                throw exceptions::AnalysisException("get_last_address(): basic block has no instructions");
            }

            std::string &get_name()
            {
                if (!name.empty())
                    return name;

                std::stringstream s;

                if (start_block)
                    name = "BB.(start block)";
                else if (end_block)
                    name = "BB.(end block)";
                else
                {
                    s << "BB." << get_first_address()
                      << "-" << get_last_address();
                    name = s.str();
                }

                return name;
            }

            /// @brief Is the current block a try-block?
            /// @return true in case this is a try block
            bool is_try_block() const
            {
                return try_block;
            }

            /// @brief Set the block is a try block
            /// @param try_block new value
            void set_try_block(bool try_block)
            {
                this->try_block = try_block;
            }

            /// @brief Is the current block a catch-block?
            /// @return true in case this is a catch block
            bool is_catch_block() const
            {
                return catch_block;
            }

            /// @brief Set the block is a catch block
            /// @param catch_block new value
            void set_catch_block(bool catch_block)
            {
                this->catch_block = catch_block;
            }

            /// @brief Get the type of handler in case is a catch block
            /// @return handler type
            DVMType *get_handler_type()
            {
                return handler_type;
            }

            /// @brief Set a handler type
            /// @param handler handler type
            void set_handler_type(DVMType *handler)
            {
                handler_type = handler;
            }

            /// @brief Get if current block is a starting block
            /// @return true in case this is a starting block
            bool is_start_block() const
            {
                return start_block;
            }

            /// @brief Set the block as a start block
            /// @param start_block new value
            void set_start_block(bool start_block)
            {
                this->start_block = start_block;
            }

            /// @brief Get if current block is an end block
            /// @return true in case this is an end block
            bool is_end_block() const
            {
                return end_block;
            }

            /// @brief Set the block as an end block
            /// @param end_block new value
            void set_end_block(bool end_block)
            {
                this->end_block = end_block;
            }
        };

        /// @brief Class to keep all the Dalvik Basic Blocks from a method
        class BasicBlocks
        {
        public:
            /// @brief blocks that are connected with others
            using connected_blocks_t = std::unordered_map<DVMBasicBlock *,
                                                          std::set<DVMBasicBlock *>>;
            /// @brief edges between nodes
            using edges_t = std::vector<std::pair<DVMBasicBlock *, DVMBasicBlock *>>;

            /// @brief Type of a node
            enum node_type_t
            {
                JOIN_NODE = 0, // len(predecessors) > 1
                BRANCH_NODE,   // len(sucessors) > 1
                REGULAR_NODE,  // other cases
            };

            /// @brief Iterator for going through a list of basic blocks in order
            using nodesiterator_t = std::vector<DVMBasicBlock *>::iterator;
            /// @brief Iterator for going through a list of basic blocks in reverse order
            using reversenodesiterator_t = std::vector<DVMBasicBlock *>::reverse_iterator;

            /// @brief Iterator for going throw a list of sucessors or predecessors
            using nodesetiterator_t = std::set<DVMBasicBlock *>::iterator;
            /// @brief Iterator for going throw a list of sucessors or predecessors in reverse order
            using reversenodesetiterator_t = std::set<DVMBasicBlock *>::reverse_iterator;

            /// @brief Iterator for going through the edges
            using edgesiterator_t = edges_t::iterator;
            /// @brief Iterator for going through the edges in reverse order
            using reverseedgesiterator_t = edges_t::reverse_iterator;

        private:
            /// @brief all the basic blocks from a method
            std::vector<DVMBasicBlock *> nodes_;

            /// @brief set of nodes that are predecessors of a node
            connected_blocks_t predecessors_;

            /// @brief set of nodes that are sucessors of a node
            connected_blocks_t sucessors_;

            /// @brief edges in the graph, this is a directed graph
            edges_t edges_;

            nodesiterator_t nodes_begin() { return nodes_.begin(); }
            nodesiterator_t nodes_end() { return nodes_.end(); }
            reversenodesiterator_t nodes_rbegin() { return nodes_.rbegin(); }
            reversenodesiterator_t nodes_rend() { return nodes_.rend(); }

            edgesiterator_t edges_begin() { return edges_.begin(); }
            edgesiterator_t edges_end() { return edges_.end(); }
            reverseedgesiterator_t edges_rbegin() { return edges_.rbegin(); }
            reverseedgesiterator_t edges_rend() { return edges_.rend(); }

        public:
            iterator_range<nodesiterator_t> nodes() { return make_range(nodes_begin(), nodes_end()); }
            iterator_range<reversenodesiterator_t> reverse_nodes() { return make_range(nodes_rbegin(), nodes_rend()); }

            iterator_range<edgesiterator_t> edges() { return make_range(edges_begin(), edges_end()); }
            iterator_range<reverseedgesiterator_t> reverse_edges() { return make_range(edges_rbegin(), edges_rend()); }

            iterator_range<nodesetiterator_t> successors(DVMBasicBlock *node)
            {
                if (sucessors_.find(node) == sucessors_.end())
                    throw exceptions::AnalysisException("analysis.hpp::sucessors: given node has no sucessors");

                return make_range(sucessors_[node].begin(), sucessors_[node].end());
            }

            iterator_range<nodesetiterator_t> predecessors(DVMBasicBlock *node)
            {
                if (predecessors_.find(node) == predecessors_.end())
                    throw exceptions::AnalysisException("analysis.hpp::sucessors: given node has no predecessors");

                return make_range(predecessors_[node].begin(), predecessors_[node].end());
            }

            iterator_range<reversenodesetiterator_t> reverse_sucessors(DVMBasicBlock *node)
            {
                if (sucessors_.find(node) == sucessors_.end())
                    throw exceptions::AnalysisException("analysis.hpp::sucessors: given node has no sucessors");

                return make_range(sucessors_[node].rbegin(), sucessors_[node].rend());
            }

            iterator_range<reversenodesetiterator_t> reverse_predecessors(DVMBasicBlock *node)
            {
                if (predecessors_.find(node) == predecessors_.end())
                    throw exceptions::AnalysisException("analysis.hpp::sucessors: given node has no predecessors");

                return make_range(predecessors_[node].rbegin(), predecessors_[node].rend());
            }

        public:
            BasicBlocks() = default;

            /// avoid any kind of copy constructor
            BasicBlocks(const BasicBlocks &temp_obj) = delete;
            BasicBlocks &operator=(const BasicBlocks &temp_obj) = delete;

            /// @brief Destructor of the BasicBlocks, we need
            /// to free the memory
            ~BasicBlocks()
            {
                // free memory
                for (auto p : nodes_)
                    delete p;
                // clear de vector of nodes
                nodes_.clear();
            }

            /// @brief Return the number of basic blocks in the graph
            /// @return number of basic blocks
            size_t get_number_of_basic_blocks() const
            {
                return nodes_.size();
            }

            /// @brief Get all predecessors from all the blocks
            /// @return constant reference to predecessors
            const connected_blocks_t &get_predecessors() const
            {
                return predecessors_;
            }

            /// @brief Get all predecessors from all the blocks
            /// @return reference to predecessors
            connected_blocks_t &get_predecessors()
            {
                return predecessors_;
            }

            /// @brief Add a node to the list of predecessors of another
            /// @param node node to add predecessor
            /// @param pred predecessor node
            void add_predecessor(DVMBasicBlock *node, DVMBasicBlock *pred)
            {
                predecessors_[node].insert(pred);
            }

            /// @brief Get all sucessors from all the blocks
            /// @return constant reference to sucessors
            const connected_blocks_t &get_sucessors() const
            {
                return sucessors_;
            }

            /// @brief Get all sucessors from all the blocks
            /// @return reference to sucessors
            connected_blocks_t &get_sucessors()
            {
                return sucessors_;
            }

            /// @brief Add a node to the list of sucessors of another
            /// @param node node to add sucessor
            /// @param suc sucessor node
            void add_sucessor(DVMBasicBlock *node, DVMBasicBlock *suc)
            {
                sucessors_[node].insert(suc);
            }

            /// @brief Add a node to the vector of nodes, we will transfer the
            /// ownership
            /// @param node node to push into our vector
            void add_node(DVMBasicBlock *node)
            {
                if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end())
                    nodes_.push_back(node);
            }

            /// @brief Add an edge to the basic blocks
            /// @param src source node
            /// @param dst edge node
            void add_edge(DVMBasicBlock *src, DVMBasicBlock *dst)
            {
                add_node(src);
                add_node(dst);
                // now insert the edge
                auto edge_pair = std::make_pair(src, dst);
                if (std::find(edges_.begin(), edges_.end(), edge_pair) == edges_.end())
                    edges_.push_back(edge_pair);
                /// now add the sucessors and predecessors
                add_sucessor(src, dst);
                add_predecessor(dst, src);
            }

            /// @brief Get a constant reference to the edges of the graph
            /// @return constant reference to the edges
            const edges_t &get_edges() const
            {
                return edges_;
            }

            /// @brief Get a reference to the edges of the graph
            /// @return reference to the edges
            edges_t &get_edges()
            {
                return edges_;
            }

            /// @brief Get the node type between JOIN_NODE, BRANCH_NODE or REGULAR_NODE
            /// @param node node to check
            /// @return type of node
            node_type_t get_node_type(DVMBasicBlock *node)
            {
                if (predecessors_[node].size() > 1)
                    return JOIN_NODE;
                else if (sucessors_[node].size() > 1)
                    return BRANCH_NODE;
                else
                    return REGULAR_NODE;
            }

            /// @brief Remove a node from the graph, this operation can
            /// be expensive on time
            /// @param node node to remove
            void remove_node(DVMBasicBlock *node);

            /// @brief Get a basic block given an idx, the idx can be one
            /// address from the first to the last address of the block
            /// @param idx address of the block to retrieve
            /// @return block that contains an instruction in that address
            DVMBasicBlock *get_basic_block_by_idx(std::uint64_t idx);

            /// @brief Get a constant reference to the nodes of the graph
            /// @return constant reference to basic blocks
            const std::vector<DVMBasicBlock *> &get_nodes() const
            {
                return nodes_;
            }

            /// @brief Get a reference to the nodes of the graph
            /// @return reference to basic blocks
            std::vector<DVMBasicBlock *> &get_nodes()
            {
                return nodes_;
            }
        };

        /// @brief specification of a field analysis
        class FieldAnalysis
        {
            /// @brief Encoded field that contains the information of the Field
            EncodedField *field;
            /// @brief name of the field
            std::string &name;
            /// @brief xrefs where field is read
            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> xrefread;
            /// @brief xrefs where field is written
            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> xrefwrite;

        public:
            FieldAnalysis(EncodedField *field)
                : field(field), name(field->get_field()->get_name())
            {
            }

            EncodedField *get_field()
            {
                return field;
            }

            std::string &get_name()
            {
                return name;
            }

            const std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefread() const
            {
                return xrefread;
            }

            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefread()
            {
                return xrefread;
            }

            const std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefwrite() const
            {
                return xrefwrite;
            }

            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefwrite()
            {
                return xrefwrite;
            }

            /// @brief Add a cross reference where the field is read in code
            /// @param c class where is read
            /// @param m method where is read
            /// @param offset idx where is read
            void add_xrefread(ClassAnalysis *c, MethodAnalysis *m, std::uint64_t offset)
            {
                xrefread.push_back(std::make_tuple(c, m, offset));
            }

            /// @brief Add a cross reference where the field is written in code
            /// @param c class where is written
            /// @param m method where is written
            /// @param offset idx where is written
            void add_xrefwrite(ClassAnalysis *c, MethodAnalysis *m, std::uint64_t offset)
            {
                xrefwrite.push_back(std::make_tuple(c, m, offset));
            }
        };

        /// @brief specification of a string analysis
        class StringAnalysis
        {
            /// @brief Value of the string
            std::string &value;
            /// @brief xref where the string is used
            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> xreffrom;

        public:
            StringAnalysis(std::string &value) : value(value)
            {
            }

            const std::string &get_value() const
            {
                return value;
            }

            const std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xreffrom() const
            {
                return xreffrom;
            }

            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xreffrom()
            {
                return xreffrom;
            }

            /// @brief Add a cross reference where the string is read
            /// @param c class where is read
            /// @param m method where is read
            /// @param offset offset where is read
            void add_xreffrom(ClassAnalysis *c, MethodAnalysis *m, std::uint64_t offset)
            {
                xreffrom.push_back(std::make_tuple(c, m, offset));
            }
        };

        /// @brief Specification of the method analysis, a method contains
        /// instructions, exceptions data, and so on...
        class MethodAnalysis
        {
        public:
            /// @brief vector of known apis of Android
            const std::vector<std::string> known_apis{
                "Landroid/", "Lcom/android/internal/util", "Ldalvik/", "Ljava/", "Ljavax/", "Lorg/apache/",
                "Lorg/json/", "Lorg/w3c/dom/", "Lorg/xml/sax", "Lorg/xmlpull/v1/", "Ljunit/", "Landroidx/"};

        private:
            /// @brief Internal method is external or internal?
            bool is_external = false;

            /// @brief encoded method or external method
            std::variant<
                EncodedMethod *,
                ExternalMethod *>
                method_encoded;

            /// @brief name of the method, store it to avoid
            /// asking one again, and again, and again
            mutable std::string name;

            /// @brief descriptor of the method
            mutable std::string descriptor;

            /// @brief access flags of the method
            /// in string format
            mutable std::string access_flag;

            /// @brief class name
            mutable std::string class_name;

            /// @brief full name of the method
            mutable std::string full_name;

            /// @brief number of registers from the method
            std::uint16_t regs_from_method;

            /// @brief number of parameters
            std::uint16_t num_of_params;

            /// @brief Instructions of the current method
            std::vector<std::unique_ptr<Instruction>> instructions_;

            /// @brief BasicBlocks from the method
            BasicBlocks basic_blocks;

            /// @brief exceptions from the method
            std::vector<Disassembler::exceptions_data> exceptions;

            /// @brief methods where current method is read
            std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> xrefread;
            /// @brief methods where method is written to
            std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> xrefwrite;

            /// @brief methods called from current method
            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> xrefto;
            /// @brief methods where this method is called
            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> xreffrom;

            /// @brief new instance of the method
            std::vector<std::pair<ClassAnalysis *, std::uint64_t>> xrefnewinstance;
            /// @brief use of const class
            std::vector<std::pair<ClassAnalysis *, std::uint64_t>> xrefconstclass;

            /// @brief Pretty print an instruction and its opcodes in a dot format to an output dot file
            /// @param dot_file file where to dump the instruction
            /// @param instr instruction to dump to dot file
            void dump_instruction_dot(std::ofstream &dot_file, Instruction *instr);

            /// @brief Pretty print a basic block in a dot graph
            /// @param dot_file file where to dump the basic block
            /// @param bb basic block to dump to dot file
            void dump_block_dot(std::ofstream &dot_file, DVMBasicBlock *bb);

            /// @brief Pretty print a method in a dot graph
            /// @param dot_file file where to dump the basic block
            /// @param name name of the dot file
            void dump_method_dot(std::ofstream &dot_file);

            /// @brief Some kind of magic function that will take all
            /// the instructions from the method, and after some wololo
            /// will generate the basic blocks.
            void create_basic_blocks();

            uniqueinstructioniterator_t instructions_begin() { return instructions_.begin(); }
            uniqueinstructioniterator_t instructions_end() { return instructions_.end(); }

            reverseuniqueinstructioniterator_t instructions_rbegin() { return instructions_.rbegin(); }
            reverseuniqueinstructioniterator_t instructions_rend() { return instructions_.rend(); }

        public:
            MethodAnalysis(
                std::variant<EncodedMethod *, ExternalMethod *> method_encoded,
                std::vector<std::unique_ptr<Instruction>> &instructions) : method_encoded(method_encoded), instructions_(std::move(instructions))
            {
                is_external = method_encoded.index() == 0 ? false : true;

                if (!is_external)
                {
                    auto em = std::get<EncodedMethod *>(method_encoded);

                    regs_from_method = em->get_code_item().get_registers_size();

                    num_of_params = em->getMethodID()->get_proto()->get_parameters().size();
                }

                if (this->instructions_.size() > 0)
                    create_basic_blocks();
            }

            /// @brief Dump the method as a dot file into
            /// the current path
            /// @param file_path reference to a path where
            /// to dump the content
            void dump_dot_file(std::string &file_path)
            {
                std::ofstream dot_file;

                dot_file.open(file_path);

                dump_method_dot(dot_file);
            }

            /// @brief Check if the method is external
            /// @return method external
            bool external() const
            {
                return is_external;
            }

            const BasicBlocks &get_basic_blocks() const
            {
                return basic_blocks;
            }

            BasicBlocks &get_basic_blocks()
            {
                return basic_blocks;
            }

            std::uint32_t get_number_of_registers() const
            {
                return regs_from_method;
            }

            std::uint32_t get_number_of_parameters() const
            {
                return num_of_params;
            }

            /// @brief Check if current method is an android api
            /// @return is android api method
            bool is_android_api() const;

            const std::string &get_name() const;

            const std::string &get_descriptor() const;

            const std::string &get_access_flags() const;

            const std::string &get_class_name() const;

            const std::string &get_full_name() const;

            std::vector<std::unique_ptr<Instruction>> &get_instructions()
            {
                return instructions_;
            }

            /// @brief Retrieve a pointer to an instruction by a given address
            /// @param addr address of the instruction to retrieve
            /// @return pointer to instruction or nullptr
            Instruction *get_instruction_by_addr(std::uint64_t addr)
            {
                auto it = std::ranges::find_if(instructions_, [=](std::unique_ptr<Instruction> &instr)
                                               { return instr->get_address() == addr; });

                if (it == instructions_.end())
                    return nullptr;

                return it->get();
            }

            /// @brief Get a range for going forward through the instructions
            /// @return forward range of instructions
            iterator_range<uniqueinstructioniterator_t> instructions() { return make_range(instructions_begin(), instructions_end()); }

            /// @brief Get a range for going backward through the instructions
            /// @return backward range of instructions
            iterator_range<reverseuniqueinstructioniterator_t> reverse_instructions() { return make_range(instructions_rbegin(), instructions_rend()); }

            std::variant<EncodedMethod *, ExternalMethod *> get_encoded_method() const
            {
                return method_encoded;
            }

            const std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> &
            get_xrefread() const
            {
                return xrefread;
            }

            std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> &
            get_xrefread()
            {
                return xrefread;
            }

            const std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> &
            get_xrefwrite() const
            {
                return xrefwrite;
            }

            std::vector<std::tuple<ClassAnalysis *, FieldAnalysis *, std::uint64_t>> &
            get_xrefwrite()
            {
                return xrefwrite;
            }

            const std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefto() const
            {
                return xrefto;
            }

            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xrefto()
            {
                return xrefto;
            }

            const std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xreffrom() const
            {
                return xreffrom;
            }

            std::vector<std::tuple<ClassAnalysis *, MethodAnalysis *, std::uint64_t>> &
            get_xreffrom()
            {
                return xreffrom;
            }

            const std::vector<std::pair<ClassAnalysis *, std::uint64_t>> &
            get_xrefnewinstance() const
            {
                return xrefnewinstance;
            }

            std::vector<std::pair<ClassAnalysis *, std::uint64_t>> &
            get_xrefnewinstance()
            {
                return xrefnewinstance;
            }

            const std::vector<std::pair<ClassAnalysis *, std::uint64_t>> &
            get_xrefconstclass() const
            {
                return xrefconstclass;
            }

            std::vector<std::pair<ClassAnalysis *, std::uint64_t>> &
            get_xrefconstclass()
            {
                return xrefconstclass;
            }

            void add_xrefread(ClassAnalysis *c, FieldAnalysis *f, std::uint64_t offset)
            {
                xrefread.push_back(std::make_tuple(c, f, offset));
            }

            void add_xrefwrite(ClassAnalysis *c, FieldAnalysis *f, std::uint64_t offset)
            {
                xrefwrite.push_back(std::make_tuple(c, f, offset));
            }

            void add_xrefto(ClassAnalysis *c, MethodAnalysis *m, std::uint64_t offset)
            {
                xrefto.push_back(std::make_tuple(c, m, offset));
            }

            void add_xreffrom(ClassAnalysis *c, MethodAnalysis *m, std::uint64_t offset)
            {
                xreffrom.push_back(std::make_tuple(c, m, offset));
            }

            void add_xrefnewinstance(ClassAnalysis *c, std::uint64_t offset)
            {
                xrefnewinstance.push_back(std::make_pair(c, offset));
            }

            void add_xrefconstclass(ClassAnalysis *c, std::uint64_t offset)
            {
                xrefconstclass.push_back(std::make_pair(c, offset));
            }
        };

        /// @brief Specification of the class analysis, this class contains
        /// fields, strings, methods...
        class ClassAnalysis
        {
        public:
            using classxref = std::unordered_map<ClassAnalysis *,
                                                 std::set<std::tuple<TYPES::REF_TYPE, MethodAnalysis *, std::uint64_t>>>;

        private:
            /// @brief definition of the class, it can be a class
            /// from the dex or an external class
            std::variant<ClassDef *, ExternalClass *> class_def;

            /// @brief is an external class
            bool is_external;

            /// @brief name of the class that it extends
            mutable std::string extends_;

            /// @brief name of the class
            mutable std::string name_;

            /// @brief map for mapping method by name with MethodAnalysis
            std::unordered_map<std::string, MethodAnalysis *> methods;
            /// @brief map for mapping EncodedField and FieldAnalysis
            std::unordered_map<EncodedField *, std::unique_ptr<FieldAnalysis>> fields;

            /// @brief Classes that this class calls
            classxref xrefto;
            /// @brief Classes that call this class
            classxref xreffrom;

            /// @brief New instances of this class
            std::vector<std::pair<MethodAnalysis *, std::uint64_t>> xrefnewinstance;

            /// @brief use of const class of this class
            std::vector<std::pair<MethodAnalysis *, std::uint64_t>> xrefconstclass;

        public:
            ClassAnalysis(std::variant<ClassDef *, ExternalClass *> class_def) : class_def(class_def)
            {
                is_external = class_def.index() == 0 ? false : true;
            }

            /// @brief add a method to the current class
            /// @param method_analysis method to include in the class
            void add_method(MethodAnalysis *method_analysis);

            size_t get_nb_methods() const
            {
                return methods.size();
            }

            size_t get_nb_fields() const
            {
                return fields.size();
            }

            /// @brief Get the class definition object
            /// @return ClassDef* or ExternalClass* object
            std::variant<ClassDef *, ExternalClass *> get_class_definition() const
            {
                return class_def;
            }

            /// @brief Is the current class an external class?
            /// @return class is external
            bool is_class_external() const
            {
                return is_external;
            }

            std::string &extends() const;

            std::string &name() const;

            /// @brief Return a vector of implemented interfaces, in
            /// the case of external class raise exception
            /// @return implemented interfaces
            std::vector<DVMClass *> &implements();

            /// @brief get a constant reference to the methods
            /// @return constant reference to methods
            const std::unordered_map<std::string, MethodAnalysis *> &
            get_methods() const
            {
                return methods;
            }

            /// @brief get a reference to the methods
            /// @returns reference to methods
            std::unordered_map<std::string, MethodAnalysis *> &
            get_methods()
            {
                return methods;
            }

            const std::unordered_map<EncodedField *, std::unique_ptr<FieldAnalysis>> &
            get_fields() const
            {
                return fields;
            }

            std::unordered_map<EncodedField *, std::unique_ptr<FieldAnalysis>> &
            get_fields()
            {
                return fields;
            }

            /// @brief Given an Encoded or ExternalMethod returns a MethodAnalysis pointer
            /// @param method method to look for
            /// @return MethodAnalysis pointer
            MethodAnalysis *get_method_analysis(std::variant<EncodedMethod *, ExternalMethod *> method);

            /// @brief Given an encoded field return a FieldAnalysis pointer
            /// @param field field to look for
            /// @return FieldAnalysis pointer
            FieldAnalysis *get_field_analysis(EncodedField *field);

            void add_field_xref_read(MethodAnalysis *method,
                                     ClassAnalysis *classobj,
                                     EncodedField *field,
                                     std::uint64_t off)
            {
                if (fields.find(field) == fields.end())
                    fields[field] = std::make_unique<FieldAnalysis>(field);
                fields[field]->add_xrefread(classobj, method, off);
            }

            void add_field_xref_write(MethodAnalysis *method,
                                      ClassAnalysis *classobj,
                                      EncodedField *field,
                                      std::uint64_t off)
            {
                if (fields.find(field) == fields.end())
                    fields[field] = std::make_unique<FieldAnalysis>(field);
                fields[field]->add_xrefwrite(classobj, method, off);
            }

            void add_method_xref_to(MethodAnalysis *method1,
                                    ClassAnalysis *classobj,
                                    MethodAnalysis *method2,
                                    std::uint64_t off)
            {
                auto method_key = method1->get_full_name();

                if (methods.find(method_key) == methods.end())
                    add_method(method1);
                methods[method_key]->add_xrefto(classobj, method2, off);
            }

            void add_method_xref_from(MethodAnalysis *method1,
                                      ClassAnalysis *classobj,
                                      MethodAnalysis *method2,
                                      std::uint64_t off)
            {
                auto method_key = method1->get_full_name();

                if (methods.find(method_key) == methods.end())
                    add_method(method1);
                methods[method_key]->add_xreffrom(classobj, method2, off);
            }

            void add_xref_to(TYPES::REF_TYPE ref_kind,
                             ClassAnalysis *classobj,
                             MethodAnalysis *methodobj,
                             std::uint64_t offset)
            {
                xrefto[classobj].insert(std::make_tuple(ref_kind, methodobj, offset));
            }

            void add_xref_from(TYPES::REF_TYPE ref_kind,
                               ClassAnalysis *classobj,
                               MethodAnalysis *methodobj,
                               std::uint64_t offset)
            {
                xreffrom[classobj].insert(std::make_tuple(ref_kind, methodobj, offset));
            }

            void add_xref_new_instance(MethodAnalysis *methodobj, std::uint64_t offset)
            {
                xrefnewinstance.push_back(std::make_pair(methodobj, offset));
            }

            void add_xref_const_class(MethodAnalysis *methodobj, std::uint64_t offset)
            {
                xrefconstclass.push_back(std::make_pair(methodobj, offset));
            }

            const classxref &
            get_xrefto() const
            {
                return xrefto;
            }

            classxref &
            get_xrefto()
            {
                return xrefto;
            }

            const classxref &
            get_xreffrom() const
            {
                return xreffrom;
            }

            classxref &
            get_xreffrom()
            {
                return xreffrom;
            }

            const std::vector<std::pair<MethodAnalysis *, std::uint64_t>>
            get_xrefnewinstance() const
            {
                return xrefnewinstance;
            }

            std::vector<std::pair<MethodAnalysis *, std::uint64_t>>
            get_xrefnewinstance()
            {
                return xrefnewinstance;
            }

            const std::vector<std::pair<MethodAnalysis *, std::uint64_t>>
            get_xrefconstclass() const
            {
                return xrefconstclass;
            }

            std::vector<std::pair<MethodAnalysis *, std::uint64_t>>
            get_xrefconstclass()
            {
                return xrefconstclass;
            }
        };
    } // namespace DEX
} // namespace KUNAI

#endif