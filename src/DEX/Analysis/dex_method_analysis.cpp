#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * MethodAnalysis methods
         */

        /**
         * @brief Constructor of MethodAnalysis it will initialize
         * various variables.
         * @param method_encoded: std::shared_ptr<ParentMethod>
         * @param dalvik_opcodes: std::shared_ptr<DalvikOpcodes> object.
         * @param instructions: std::map<std::uint64_t, std::shared_ptr<Instruction>> all the DEX instructions.
         * @return void.
         */
        MethodAnalysis::MethodAnalysis(std::shared_ptr<ParentMethod> method_encoded, std::shared_ptr<DalvikOpcodes> dalvik_opcodes, std::map<std::uint64_t, std::shared_ptr<Instruction>> instructions)
        {
            this->is_external = method_encoded->is_external();

            this->method_encoded = method_encoded;
            this->dalvik_opcodes = dalvik_opcodes;
            this->instructions = instructions;
            this->exceptions = std::make_shared<Exception>();

            if (this->instructions.size() > 0)
                this->create_basic_block();
        }

        /**
         * @brief MethodAnalysis destructor.
         * @return void.
         */
        MethodAnalysis::~MethodAnalysis() {}

        /**
         * @brief return if the method is instance of std::shared_ptr<ExternalMethod>
         * @return bool
         */
        bool MethodAnalysis::external()
        {
            return is_external;
        }

        /**
         * @brief check if current method is an Android API.
         * @return bool
         */
        bool MethodAnalysis::is_android_api()
        {
            if (!is_external)
                return false;

            std::string class_name = this->class_name();

            for (auto it = known_apis.begin(); it != known_apis.end(); it++)
            {
                if (class_name.find(*it) == 0)
                    return true;
            }

            return false;
        }

        /**
         * @brief Return method_encoded object, this can
         * be of different types EncodedMethod or ExternalMethod
         * must check which one it is.
         * @return std::shared_ptr<ParentMethod>
         */
        std::shared_ptr<ParentMethod> MethodAnalysis::get_method()
        {
            return method_encoded;
        }

        /**
         * @brief return the method name.
         * @return std::string
         */
        std::string MethodAnalysis::name()
        {
            std::string name;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                name = method->get_name();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                name = *method->get_method()->get_method_name();
            }

            return name;
        }

        /**
         * @brief return method prototype (descriptor)
         * @return std::string
         */
        std::string MethodAnalysis::descriptor()
        {
            std::string descriptor;
            
            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                descriptor = method->get_descriptor();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                descriptor = method->get_method()->get_method_prototype()->get_proto_str();
            }

            return descriptor;
        }

        /**
         * @brief return access as string.
         * @return std::string
         */
        std::string MethodAnalysis::access()
        {
            std::string access_flag;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                access_flag = this->dalvik_opcodes->get_access_flags_string(method->get_access_flags());
            }

            return access_flag;
        }

        /**
         * @brief get the class name from the method.
         * @return std::string
         */
        std::string MethodAnalysis::class_name()
        {
            std::string class_name;

            if (is_external)
            {
                std::shared_ptr<ExternalMethod> method = std::dynamic_pointer_cast<ExternalMethod>(method_encoded);
                class_name = method->get_class_name();
            }
            else
            {
                std::shared_ptr<EncodedMethod> method = std::dynamic_pointer_cast<EncodedMethod>(method_encoded);
                class_name = method->get_method()->get_method_class()->get_raw();
            }

            return class_name;
        }

        /**
         * @brief get whole name with class name, method name and descriptor.
         * @return std::string
         */
        std::string MethodAnalysis::full_name()
        {
            std::string class_name = this->class_name();
            std::string descriptor = this->descriptor();
            std::string name = this->name();

            return class_name + " " + name + " " + descriptor;
        }

        /**
         * @brief Insert a new xref of method reading.
         * @param class_object: ClassAnalysis where the method is read.
         * @param field_object: FieldAnalysis maybe where the method is read from... Dunno
         * @param offset: offset of the instruction where the method is read.
         * @return void
         */
        void MethodAnalysis::add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, field_object, offset});
        }

        /**
         * @brief Insert a new xref of method written.
         * @param class_object: ClassAnalysis where the method is written.
         * @param field_object: FieldAnalysis maybe where the method is written... Dunno
         * @param offset: offset of the instruction where the method is written.
         * @return void
         */
        void MethodAnalysis::add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<FieldAnalysis> field_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, field_object, offset});
        }

        /**
         * @brief Return all the xref where method is read, with or without offset.
         * 
         * @return std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>> 
         */
        const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>>& MethodAnalysis::get_xref_read()
        {
            return xrefread;
        }

        /**
         * @brief Return all the xref where method is written, with or without offset.
         * @param withoffset: return tuple with or without offset.
         * @return if withoffset == true `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<FieldAnalysis>, std::uint64_t>>'
         *         if withoffset == false `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<FieldAnalysis>>>'
         */
        const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::FieldAnalysis>, uint64_t>>& MethodAnalysis::get_xref_write()
        {
            return xrefwrite;
        }

        /**
         * @brief Add a reference to a method called by this method.
         * @param class_object: std::shared_ptr<ClassAnalysis> class of the method called.
         * @param method_object: std::shared_ptr<MethodAnalysis> method called from current method.
         * @param offset: std::uint64_t offset where call is done.
         * @return void
         */
        void MethodAnalysis::add_xref_to(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefto.push_back({class_object, method_object, offset});
        }

        /**
         * @brief Add a reference of a method that calls current method.
         * @param class_object: std::shared_ptr<ClassAnalysis> class of the method that calls current method.
         * @param method_object: std::shared_ptr<MethodAnalysis> method that calls current method.
         * @param offset: std::uint64_t offset where call is done.
         * @return void
         */
        void MethodAnalysis::add_xref_from(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xreffrom.push_back({class_object, method_object, offset});
        }

        /**
         * @brief get the methods where current method is called, with or without offset.
         * @param withoffset: bool return offsets or not.
         * @return if withoffset == true `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>'
         *         if withoffset == false `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>>>'
         */

        /**
         * @brief get the methods where current method is called, with or without offset.
         * 
         * @return const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>& 
         */
        const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>& MethodAnalysis::get_xref_to()
        {
            return xrefto;
        }

        /**
         * @brief get the methods called by current method, with or without offset.
         * @param withoffset: bool return offsets or not.
         * @return if withoffset == true `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>, std::uint64_t>>'
         *         if withoffset == false `std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::shared_ptr<MethodAnalysis>>>'
         */


        /**
         * @brief get the methods called by current method, with or without offset.
         * 
         * @return const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>& 
         */
        const std::vector<std::tuple<std::shared_ptr<KUNAI::DEX::ClassAnalysis>, std::shared_ptr<KUNAI::DEX::MethodAnalysis>, uint64_t>>& MethodAnalysis::get_xref_from()
        {
            return xreffrom;
        }

        /**
         * @brief Add a cross reference to another class that is instanced within this method.
         * @param class_object: std::shared_ptr<ClassAnalysis> class_object instanced class.
         * @param offset: std::uint64_t offset of the method
         * @return void
         */
        void MethodAnalysis::add_xref_new_instance(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset)
        {
            xrefnewinstance.push_back({class_object, offset});
        }

        /**
         * @brief Add a cross reference to another classtype.
         * @param class_object: std::shared_ptr<ClassAnalysis>
         * @param offset: std::uint64_t
         * @return void
         */
        void MethodAnalysis::add_xref_const_class(std::shared_ptr<ClassAnalysis> class_object, std::uint64_t offset)
        {
            xrefconstclass.push_back({class_object, offset});
        }

        /**
         * @brief return the cross references of classes instanced by this method.
         * @return std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>>
         */
        std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> MethodAnalysis::get_xref_new_instance()
        {
            return xrefnewinstance;
        }

        /**
         * @brief return all the cross references of another classtype.
         * @return std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>>
         */
        std::vector<std::tuple<std::shared_ptr<ClassAnalysis>, std::uint64_t>> MethodAnalysis::get_xref_const_class()
        {
            return xrefconstclass;
        }

        /**
         * @brief get the instructions from the method.
         * @return std::map<std::uint64_t, std::shared_ptr<Instruction>>
         */
        std::map<std::uint64_t, std::shared_ptr<Instruction>> MethodAnalysis::get_instructions()
        {
            return instructions;
        }

        /**
         * @brief get the basic blocks with the DVMBasicBlocks with the instructions.
         * @return std::shared_ptr<BasicBlocks>
         */
        std::shared_ptr<BasicBlocks> MethodAnalysis::get_basic_blocks()
        {
            return basic_blocks;
        }

        /**
         * @brief Get all the exceptions from the method.
         * @return std::shared_ptr<Exception>
         */
        std::shared_ptr<Exception> MethodAnalysis::get_exceptions()
        {
            return exceptions;
        }

        /**
         * @brief method to create basic blocks for the method.
         * @return void
         */
        void MethodAnalysis::create_basic_block()
        {
            basic_blocks = std::make_shared<BasicBlocks>();
            std::shared_ptr<DVMBasicBlock> current_basic = std::make_shared<DVMBasicBlock>(0, dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);

            // push the first basic block
            basic_blocks->push_basic_block(current_basic);

            std::vector<std::int64_t> l;
            std::map<std::uint64_t, std::vector<std::int64_t>> h;

            std::cout << "Parsing the instructions for method " << std::hex << std::dynamic_pointer_cast<EncodedMethod>(method_encoded)->full_name() << std::endl;
            for (auto it = instructions.begin(); it != instructions.end(); it++)
            {
                auto idx = std::get<0>(*it);
                auto ins = std::get<1>(*it);

                if (dalvik_opcodes->get_instruction_operation(ins->get_OP()) ==
                    DVMTypes::Operation::BRANCH_DVM_OPCODE)
                {
                    auto v = determine_next(ins, idx, instructions);
                    h[idx] = v;
                    l.insert(l.end(), v.begin(), v.end());
                }
            }

            std::cout << "Parsing the exceptions" << std::endl;
            auto excepts = determine_exception(dalvik_opcodes, std::dynamic_pointer_cast<EncodedMethod>(method_encoded));
            for (auto it = excepts.begin(); it != excepts.end(); it++)
            {
                l.push_back(it->try_value_start_addr);
                for (auto handler = it->handler.begin(); handler != it->handler.end(); handler++)
                {
                    l.push_back(handler->handler_start_addr);
                }
            }

            std::cout << "Creating basic blocks" << std::endl;
            for (auto it = instructions.begin(); it != instructions.end(); it++)
            {
                auto idx = std::get<0>(*it);
                auto ins = std::get<1>(*it);

                if (std::find(l.begin(), l.end(), static_cast<std::int64_t>(idx)) != l.end())
                {
                    if (current_basic->get_nb_instructions() != 0)
                    {
                        current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);
                        basic_blocks->push_basic_block(current_basic);
                    }
                }

                current_basic->push(ins);

                if (h.find(idx) != h.end())
                {
                    current_basic = std::make_shared<DVMBasicBlock>(current_basic->get_end(), dalvik_opcodes, basic_blocks, std::dynamic_pointer_cast<EncodedMethod>(method_encoded), instructions);
                    basic_blocks->push_basic_block(current_basic);
                }
            }

            if (current_basic->get_nb_instructions() == 0)
            {
                basic_blocks->pop_basic_block();
            }

            std::cout << "Settings basic blocks childs" << std::endl;
            auto bbs = basic_blocks->get_basic_blocks();
            for (auto it = bbs.begin(); it != bbs.end(); it++)
            {
                auto bb = *it;
                bb->set_child(h[bb->get_end() - bb->get_last_length()]);
            }

            std::cout << "Creating exceptions" << std::endl;
            this->exceptions->add(excepts, this->basic_blocks);

            for (auto it = bbs.begin(); it != bbs.end(); it++)
            {
                (*it)->set_exception_analysis(this->exceptions->get_exception((*it)->get_start(), (*it)->get_end()));
            }
        }
    }
}