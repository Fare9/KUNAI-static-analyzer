#include "KUNAI/DEX/Analysis/dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * FieldAnalysis methods
         */

        FieldAnalysis::FieldAnalysis(EncodedField* field) : field(field)
        {
        }

        void FieldAnalysis::add_xref_read(ClassAnalysis* class_object, MethodAnalysis* method_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, method_object, offset});
        }

        void FieldAnalysis::add_xref_write(ClassAnalysis* class_object, MethodAnalysis* method_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, method_object, offset});
        }
    }
}