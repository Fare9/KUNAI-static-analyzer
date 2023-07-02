#include "backward-slicing/backward_analysis.hpp"

using namespace KUNAI::DEX;

int8_t BackwardAnalysis::parse_targets_file(std::string file_path)
{
    std::ifstream targetsfs;
    std::string myline;
    std::regex src_pattern("(L.+?;)->(.+?)(\\(.+\\).)");
    std::smatch match;

    targetsfs.open(file_path);
    if (targetsfs.is_open())
    {
        target_info t;
        while (targetsfs)
        {
            std::getline(targetsfs, myline);
            if (myline.length() != 0)
            {
                // Extract class, method and proto using regex
                if (std::regex_match(myline, match, src_pattern))
                {
                    t.class_name = match[1];
                    t.method_name = match[2];
                    t.prototype = match[3];
                    t.is_parsed_correct = true;
                }
                else
                {
                    t.class_name = "";
                    t.method_name = "";
                    t.prototype = "";
                    t.is_parsed_correct = false;
                }
                targets_info.push_back(t);
            }
        }
        targetsfs.close();
        return 0;
    }
    else
    {
        std::cerr << "Couldn't open file " << file_path << "\n";
        return -1;
    }
}

void BackwardAnalysis::find_parameters_registers(KUNAI::DEX::MethodAnalysis *xreffrom_method, KUNAI::DEX::MethodAnalysis *target_method, std::uint64_t addr)
{
    auto &instructions = xreffrom_method->get_instructions();

    // find invoke instruction within the instructions of the xreffrom method using the address of the xref
    auto it = std::find_if(instructions.begin(), instructions.end(), [&](std::unique_ptr<KUNAI::DEX::Instruction> &instr)
    {
        return instr->get_address() == addr;
    });

    if (it == instructions.end())
        return;
    
    auto instruc = it->get();

    // check specific invoke instruction type to obtain the method's parameters' registers
    const auto instruc_type = instruc->get_instruction_type();
    if (instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION35C)
    {
        //std::cout << "Invoke instruction in xreffrom method: " << instruc->print_instruction() << std::endl;

        KUNAI::DEX::Instruction35c *instruction35c = reinterpret_cast<KUNAI::DEX::Instruction35c *>(instruc);

        parameters_registers = instruction35c->get_registers();
        
        auto opcode = static_cast<KUNAI::DEX::TYPES::opcodes>(instruction35c->get_instruction_opcode());

        // if instruction is invoke-virtual or invoke-direct params start from the second register
        if (opcode == KUNAI::DEX::TYPES::OP_INVOKE_VIRTUAL || opcode == KUNAI::DEX::TYPES::OP_INVOKE_DIRECT)
        {
            parameters_registers.erase(parameters_registers.begin());
        }
        // else instruction is invoke-super, invoke-static or invoke-interface params start from the first register

    }
    else if (instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION45CC)
    {
        KUNAI::DEX::Instruction45cc *instruction45cc = reinterpret_cast<KUNAI::DEX::Instruction45cc *>(instruc);
        parameters_registers = instruction45cc->get_registers();
        parameters_registers.erase(parameters_registers.begin());
    }
    else if (instruc_type == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION4RCC)
    {
        KUNAI::DEX::Instruction4rcc *instruction4rcc = reinterpret_cast<KUNAI::DEX::Instruction4rcc *>(instruc);
        auto &regs = instruction4rcc->get_registers();
        for (std::size_t i = 0; i < parameters_registers.size(); ++i) 
            parameters_registers[i] = static_cast<uint8_t>(regs[i]);
        parameters_registers.erase(parameters_registers.begin());
    }
}

void BackwardAnalysis::find_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method){
    // check if registers have been identified prior to this
    if (!parameters_registers.empty()){
        std::uint8_t reg;
        auto &instructions = xreffrom_method->get_instructions();
        for (auto &instruc : instructions)
        {
            // Check for each possible write instruction type (move, const, return, binary operation)
            if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION12X)
            {
                KUNAI::DEX::Instruction12x *instruction12x = reinterpret_cast<KUNAI::DEX::Instruction12x *>(instruc.get());
                reg = instruction12x->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11N)
            {
                KUNAI::DEX::Instruction11n *instruction11n = reinterpret_cast<KUNAI::DEX::Instruction11n *>(instruc.get());
                reg = instruction11n->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22X)
            {
                KUNAI::DEX::Instruction22x *instruction22x = reinterpret_cast<KUNAI::DEX::Instruction22x *>(instruc.get());
                reg = instruction22x->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21S)
            {
                KUNAI::DEX::Instruction21s *instruction21s = reinterpret_cast<KUNAI::DEX::Instruction21s *>(instruc.get());
                reg = instruction21s->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21H)
            {
                KUNAI::DEX::Instruction21h *instruction21h = reinterpret_cast<KUNAI::DEX::Instruction21h *>(instruc.get());
                reg = instruction21h->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION21C)
            {
                KUNAI::DEX::Instruction21c *instruction21c = reinterpret_cast<KUNAI::DEX::Instruction21c *>(instruc.get());
                reg = instruction21c->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31I)
            {
                KUNAI::DEX::Instruction31i *instruction31i = reinterpret_cast<KUNAI::DEX::Instruction31i *>(instruc.get());
                reg = instruction31i->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION31C)
            {
                KUNAI::DEX::Instruction31c *instruction31c = reinterpret_cast<KUNAI::DEX::Instruction31c *>(instruc.get());
                reg = instruction31c->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION11X)
            {
                KUNAI::DEX::Instruction11x *instruction11x = reinterpret_cast<KUNAI::DEX::Instruction11x *>(instruc.get());
                reg = instruction11x->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22B)
            {
                KUNAI::DEX::Instruction22b *instruction22b = reinterpret_cast<KUNAI::DEX::Instruction22b *>(instruc.get());
                reg = instruction22b->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22S)
            {
                KUNAI::DEX::Instruction22s *instruction22s = reinterpret_cast<KUNAI::DEX::Instruction22s *>(instruc.get());
                reg = instruction22s->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION22C)
            {
                KUNAI::DEX::Instruction22c *instruction22c = reinterpret_cast<KUNAI::DEX::Instruction22c *>(instruc.get());
                reg = instruction22c->get_destination();
                
                for (auto & pr : parameters_registers){
                    if (pr == reg)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
            // Write instruction double (two destination registers)
            else if (instruc->get_instruction_type() == KUNAI::DEX::dexinsttype_t::DEX_INSTRUCTION51L)
            {
                KUNAI::DEX::Instruction51l *instruction51l = reinterpret_cast<KUNAI::DEX::Instruction51l *>(instruc.get());
                reg = instruction51l->get_first_register();
                auto reg2 = instruction51l->get_second_register();
                for (auto & pr : parameters_registers){
                    if (pr == reg || pr == reg2)
                    {
                        instruction_mapped_registers[pr] = instruc.get();
                        break;
                    }
                }
            }
        }
    }
    else
    {
        std::cerr << "Backward analysis error: Parameter registers not identified" << std::endl;
    }
}

void BackwardAnalysis::print_parameters_definition(KUNAI::DEX::MethodAnalysis *xreffrom_method){
    if (!instruction_mapped_registers.empty()){
        size_t i = 1;
        for (auto & r : parameters_registers){
            auto e = instruction_mapped_registers.find(r);
            if (e != instruction_mapped_registers.end()) {
                std::cout << "\tParameter " << i++ << " (register v" << unsigned(e->first) << "): defined in instruction " << e->second->print_instruction() << " (address " << e->second->get_address() << ")\n";
            }
            else
            {
                std::cout << "\tParameter " << i++ << " (register v" << r << "): defined in method invocation" << xreffrom_method->get_full_name() << "\n";
            }
        }
    }
    else
    {
        std::cerr << "Error: Did not perform backward analysis" << std::endl;
    }
}