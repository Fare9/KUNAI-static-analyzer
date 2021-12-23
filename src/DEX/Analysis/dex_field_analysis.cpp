#include "dex_analysis.hpp"

namespace KUNAI
{
    namespace DEX
    {
        /**
         * FieldAnalysis methods
         */

        FieldAnalysis::FieldAnalysis(std::shared_ptr<EncodedField> field) : field(field)
        {
        }

        void FieldAnalysis::add_xref_read(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefread.push_back({class_object, method_object, offset});
        }

        void FieldAnalysis::add_xref_write(std::shared_ptr<ClassAnalysis> class_object, std::shared_ptr<MethodAnalysis> method_object, std::uint64_t offset)
        {
            xrefwrite.push_back({class_object, method_object, offset});
        }
    }
}