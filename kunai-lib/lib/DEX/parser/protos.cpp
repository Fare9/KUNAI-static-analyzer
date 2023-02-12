//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file protos.cpp

#include "Kunai/DEX/parser/protos.hpp"
#include "Kunai/Utils/logger.hpp"

#include <iomanip>

using namespace KUNAI::DEX;

void ProtoID::parse_parameters(stream::KunaiStream *stream,
                               Types *types,
                               std::uint32_t parameters_off)
{
    auto current_offset = stream->tellg();
    std::uint32_t n_parameters;
    std::uint16_t type_id;

    if (!parameters_off) // no parameters?
        return;

    stream->seekg(parameters_off, std::ios_base::beg);

    // read the number of parameters
    stream->read_data<std::uint32_t>(n_parameters, sizeof(std::uint32_t));

    for (size_t I = 0; I < n_parameters; ++I)
    {
        stream->read_data<std::uint16_t>(type_id, sizeof(std::uint16_t));

        parameters.push_back(types->get_type_from_order(type_id));
    }

    stream->seekg(current_offset, std::ios_base::beg);
}

void Protos::parse_protos(stream::KunaiStream *stream,
                          std::uint32_t number_of_protos,
                          std::uint32_t offset,
                          Strings *strings,
                          Types *types)
{
    auto logger = LOGGER::logger();
    auto current_offset = stream->tellg();
    this->number_of_protos = number_of_protos;
    protoid_t protoid;
    std::uint32_t shorty_idx = 0, //! id for prototype string
        return_type_idx = 0,      //! id for type of return
        parameters_off = 0;       //! offset of parameters

    // set to current offset
    stream->seekg(offset, std::ios_base::beg);

    for (size_t I; I < number_of_protos; ++I)
    {
        stream->read_data<std::uint32_t>(shorty_idx, sizeof(std::uint32_t));

        stream->read_data<std::uint32_t>(return_type_idx, sizeof(std::uint32_t));

        stream->read_data<std::uint32_t>(parameters_off, sizeof(std::uint32_t));

        protoid = std::make_unique<ProtoID>(stream,
                                            types,
                                            strings->get_string_by_id(shorty_idx),
                                            return_type_idx,
                                            parameters_off);
        proto_ids.push_back(std::move(protoid));
    }

    stream->seekg(offset, std::ios_base::beg);
}

std::ostream &operator<<(std::ostream &os, const Protos &entry)
{
    size_t I = 0;
    const auto &protoids = entry.get_proto_ids();

    os << std::setw(30) << std::left << std::setfill(' ') << "DEX Protos:\n";

    for (const auto &protoid : protoids)
    {
        os << std::left << std::setfill(' ') << "Proto (" << I++ << "): " << protoid->get_shorty_idx() << "\n";
    }

    return os;
}

void Protos::to_xml(std::ofstream &xml_file)
{
    xml_file << "<protos>\n";

    for (const auto &protoid : proto_ids)
    {
        xml_file << "\t<proto>\n";

        xml_file << "\t\t<parameters>\n";
        for (auto param : protoid->get_parameters())
        {
            xml_file << "\t\t\t<parameter>\n";
            xml_file << "\t\t\t\t<type>" << param->print_type() << "</type>\n";
            xml_file << "\t\t\t\t<raw>" << param->get_raw() << "</raw>\n";
            xml_file << "\t\t\t</parameter>\n";
        }
        xml_file << "\t\t</parameters>\n";

        xml_file << "\t\t<return>\n";
        xml_file << "\t\t\t\t<type>" << protoid->get_return_type()->print_type() << "</type>\n";
        xml_file << "\t\t\t\t<raw>" << protoid->get_return_type()->get_raw() << "</raw>\n";
        xml_file << "\t\t</return>\n";

        xml_file << "\t</proto>\n";
    }

    xml_file << "</protos>\n";
}