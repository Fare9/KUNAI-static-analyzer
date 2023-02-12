//--------------------------------------------------------------------*- C++ -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file protos.hpp
// @brief Manage all the prototypes used in a DEX file, the prototypes
// include parameters return types, etc.

/*
 * Each proto structure is:
 *  ProtoID {
 *      uint shorty_idx, # OFFSET to a string with return type
 *      uint return_type_idx, # OFFSET to a type which points to string with return type
 *      uint parameters_off # ---------| Offset to parameters in type list type
 *  }                                  |
 *                                     |
 *  ------------------------------------
 *  |
 *  v
 *  type_list {
 *      uint size, # Size of the list in entries
 *      list type_item[size] # ---------|
 *  }                                   |
 *                                      |
 *  -------------------------------------
 *  |
 *  v
 *  type_item {
 *      ushort type_idx # type_idx of each member
 *  }
 *
 *  ProtoID[] protos
 */

#ifndef KUNAI_DEX_PARSER_PROTOS_HPP
#define KUNAI_DEX_PARSER_PROTOS_HPP

#include "Kunai/Utils/kunaistream.hpp"
#include "Kunai/DEX/parser/types.hpp"
#include "Kunai/DEX/parser/strings.hpp"

#include <vector>

namespace KUNAI
{
namespace DEX
{

    /// @brief Store the information of a ProtoID, this is
    /// a string with the return type, the list of parameters
    /// and a string with the prototype
    class ProtoID
    {
        /// @brief reference to a string with the prototype
        std::string &shorty_idx;
        /// @brief type returned by the prototype
        DVMType *return_type;
        /// @brief vector of the parameter types
        std::vector<DVMType *> parameters;
        /// @brief Parse the parameters for the ProtoID
        /// @param stream stream where to read the type ids
        /// @param types for obtaining the types
        /// @param parameters_off offset where the parameters are
        void parse_parameters(
            stream::KunaiStream *stream,
            Types *types,
            std::uint32_t parameters_off);

    public:
        /// @brief Constructor of a ProtoID, the ProtoID parses its own parameters
        /// @param stream stream to read the parameter types ids
        /// @param types types to obtain result and parameter types
        /// @param shorty_idx string with prototype
        /// @param return_type_idx id of the return type
        /// @param parameters_off offset where parameter type ids are stored
        ProtoID(
            stream::KunaiStream *stream,
            Types *types,
            std::string& shorty_idx,
            std::uint32_t return_type_idx,
            std::uint32_t parameters_off)
            : shorty_idx(shorty_idx),
              return_type(types->get_type_by_id(return_type_idx))
        {
            parse_parameters(stream, types, parameters_off);
        }

        /// @brief Get constant reference to shorty_idx string
        /// @return constant reference to shorty_idx
        const std::string& get_shorty_idx() const
        {
            return shorty_idx;
        }

        /// @brief Get a reference to shorty_idx string
        /// @return reference to shorty_idx
        std::string& get_shorty_idx()
        {
            return shorty_idx;
        }

        /// @brief Get a constant reference to the return type
        /// @return constant reference to return type
        const DVMType* get_return_type() const
        {
            return return_type;
        }

        /// @brief Get a reference to the return type
        /// @return reference to return type
        DVMType* get_return_type()
        {
            return return_type;
        }

        /// @brief Get a constant reference to the parameters
        /// @return const reference to vector with the parameters
        const std::vector<DVMType *>& get_parameters() const
        {
            return parameters;
        }

        /// @brief Get a reference to the parameters
        /// @return reference to vector with the parameters
        std::vector<DVMType*>& get_parameters()
        {
            return parameters;
        } 
    };

    using protoid_t = std::unique_ptr<ProtoID>;

    /// @brief Class to manage all the ProtoID from the 
    /// DEX file
    class Protos
    {
        /// @brief Vector with all the proto_id
        /// it has memory ownership
        std::vector<protoid_t> proto_ids;
        /// @brief nummber of protos to read
        std::uint32_t number_of_protos;

    public:
        /// @brief Default constructor of protos
        Protos() = default;
        /// @brief Default destructor of protos
        ~Protos() = default;

        /// @brief Parse all the ProtoIDs from the file
        /// @param stream stream with dex file
        /// @param number_of_protos number of protos to read
        /// @param offset offset where to read the protos
        /// @param strings object with all the strings from the dex
        /// @param types object with all the types from the dex
        void parse_protos(stream::KunaiStream* stream,
                          std::uint32_t number_of_protos,
                          std::uint32_t offset,
                          Strings* strings,
                          Types* types);

        /// @brief Return a constant reference to proto_ids vector
        /// @return const reference to the proto_id vector
        const std::vector<protoid_t>& get_proto_ids() const
        {
            return proto_ids;
        }
        
        /// @brief Return a reference to proto_ids vector
        /// @return reference to proto_ids vector
        std::vector<protoid_t>& get_proto_ids()
        {
            return proto_ids;
        }

        /// @brief get the number of proto_ids
        /// @return number of proto_ids
        std::uint32_t get_number_of_protos() const
        {
            return number_of_protos;
        }

        /// @brief Return a pretty printed version of the proto_ids
        /// @param os stream where to print it
        /// @param entry entry to print
        /// @return stream with printed data
        friend std::ostream &operator<<(std::ostream &os, const Protos &entry);
        
        /// @brief Dump the proto_ids to an XML file
        /// @param xml_file file where to dump the data
        void to_xml(std::ofstream& xml_file);
    };
} // namespace DEX
} // namespace KUNAI

#endif