// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRToOpGraph.cpp

#include "MjolnIR/Transforms/MjolnIRToOpGraph.hpp"

#include <cassert>
#include <map>
#include <optional>
#include <mlir/Pass/Pass.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/IndentedOstream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include "MjolnIR/Dalvik/MjolnIRDialect.hpp"
#include "MjolnIR/Dalvik/MjolnIRTypes.hpp"
#include "MjolnIR/Dalvik/MjolnIROps.hpp"

using namespace KUNAI::MjolnIR;

static const mlir::StringRef kLineStyleControlFlow = "dashed";
static const mlir::StringRef kLineStyleDataFlow = "solid";
static const mlir::StringRef kShapeNode = "ellipse";
static const mlir::StringRef kShapeNone = "plain";
static const std::size_t maxLabelLen = 250;

static int64_t getLargeAttributeSizeLimit()
{
    // Use the default from the printer flags if possible.
    if (std::optional<int64_t> limit =
            mlir::OpPrintingFlags().getLargeElementsAttrLimit())
        return *limit;
    return 16;
}

/// Return all values printed onto a stream as a string.
static std::string strFromOs(mlir::function_ref<void(mlir::raw_ostream &)> func)
{
    std::string buf;
    llvm::raw_string_ostream os(buf);
    func(os);
    return os.str();
}

/// Escape special characters such as '\n' and quotation marks.
static std::string escapeString(std::string str)
{
    return strFromOs([&](mlir::raw_ostream &os)
                     { os.write_escaped(str); });
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str)
{
    return "\"" + str + "\"";
}

using AttributeMap = std::map<std::string, std::string>;

/// This struct represents a node in the DOT language. Each node has an
/// identifier and an optional identifier for the cluster (subgraph) that
/// contains the node.
/// Note: In the DOT language, edges can be drawn only from nodes to nodes, but
/// not between clusters. However, edges can be clipped to the boundary of a
/// cluster with `lhead` and `ltail` attributes. Therefore, when creating a new
/// cluster, an invisible "anchor" node is created.
struct Node
{
public:
    Node(int id = 0, std::optional<int> clusterId = std::nullopt)
        : id(id), clusterId(clusterId) {}

    int id;
    std::optional<int> clusterId;
};

class PrintOpPass : public mlir::PassWrapper<PrintOpPass, mlir::OperationPass<void>>
{
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpPass)

    PrintOpPass(mlir::raw_ostream &os) : os(os) {}
    PrintOpPass(const PrintOpPass &o) : PrintOpPass(o.os.getOStream()) {}

    void runOnOperation() override
    {
        emitGraph([&]()
                  { 
                auto op = getOperation();
                processFunction(op); 
                emitAllEdgeStmts(op); });
    }

    /// @brief Generate an attribute statement as a string
    /// @param key key of the attribute
    /// @param value value of the attribute
    /// @return
    std::string attrStmt(const mlir::Twine &key, const mlir::Twine &value)
    {
        return (key + " = " + value).str();
    }

    /// Emit a cluster (subgraph). The specified builder generates the body of the
    /// cluster. Return the anchor node of the cluster.
    Node emitClusterStmt(mlir::function_ref<void()> builder, std::string label = "")
    {
        int clusterId = ++counter;
        os << "subgraph cluster_" << clusterId << " {\n";
        os.indent();
        // Emit invisible anchor node from/to which arrows can be drawn.
        Node anchorNode = emitNodeStmt(" ", kShapeNone);
        os << attrStmt("label", quoteString(escapeString(std::move(label))))
           << ";\n";
        builder();
        os.unindent();
        os << "}\n";
        return Node(anchorNode.id, clusterId);
    }

    void emitAttrList(mlir::raw_ostream &os, const AttributeMap &map)
    {
        os << "[";
        interleaveComma(map, os, [&](const auto &it)
                        { os << this->attrStmt(it.first, it.second); });
        os << "]";
    }

    // Print an MLIR attribute to `os`. Large attributes are truncated.
    void emitMlirAttr(mlir::raw_ostream &os, mlir::Attribute attr)
    {
        // A value used to elide large container attribute.
        int64_t largeAttrLimit = getLargeAttributeSizeLimit();

        // Always emit splat attributes.
        if (attr.isa<mlir::SplatElementsAttr>())
        {
            attr.print(os);
            return;
        }

        // Elide "big" elements attributes.
        auto elements = attr.dyn_cast<mlir::ElementsAttr>();
        if (elements && elements.getNumElements() > largeAttrLimit)
        {
            os << std::string(elements.getShapedType().getRank(), '[') << "..."
               << std::string(elements.getShapedType().getRank(), ']') << " : "
               << elements.getType();
            return;
        }

        auto array = attr.dyn_cast<mlir::ArrayAttr>();
        if (array && static_cast<int64_t>(array.size()) > largeAttrLimit)
        {
            os << "[...]";
            return;
        }

        // Print all other attributes.
        std::string buf;
        llvm::raw_string_ostream ss(buf);
        attr.print(ss);
        os << truncateString(ss.str());
    }

    /// @brief Emit the graph calling the functions that are part of builder
    /// @param builder functions to call during the generation of dot file.
    void emitGraph(mlir::function_ref<void()> builder)
    {
        os << "digraph G {\n";
        os.indent();
        // Edges between clusters are allowed only in compound mode.
        os << attrStmt("compound", "true") << ";\n";
        builder();
        os.unindent();
        os << "}\n";
    }

    void processFunction(mlir::Operation *op)
    {
        assert(mlir::isa<mlir::FunctionOpInterface>(op) &&
               "Operation provided must implement FunctionOpInterface.");

        auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op);

        emitClusterStmt([&]()
                        {
            for (auto &region : func->getRegions())
                processRegion(region); },
                        getLabel(op));
    }

    void processRegion(mlir::Region &region)
    {
        for (auto &bb : region.getBlocks())
        {
            emitClusterStmt([&]()
                            { processBlock(bb); },
                            "BB");
        }
    }

    /// Generate a label for a block argument.
    std::string getLabel(mlir::BlockArgument arg)
    {
        return "arg" + std::to_string(arg.getArgNumber());
    }

    void processBlock(mlir::Block &block)
    {
        /// for (auto &blockArg : block.getArguments())
        ///     emitNodeStmt(getLabel(blockArg));

        int previous = -1;
        for (auto &instr : block.getOperations())
        {
            auto node = emitNodeStmt(getLabel(&instr));

            instrToNode[&instr] = node;

            if (previous == -1)
                previous = node.id;
            else
            {
                auto current = node.id;

                os << "v" << previous;
                os << " -> ";
                os << "v" << current;
                os << ";\n";
                previous = current;
            }
        }
    }

    Node emitNodeStmt(std::string label, mlir::StringRef shape = kShapeNode)
    {
        int nodeId = ++counter;
        AttributeMap attrs;
        attrs["label"] = quoteString(escapeString(std::move(label)));
        attrs["shape"] = shape.str();
        os << "v" << nodeId;
        emitAttrList(os, attrs);
        os << ";\n";
        return Node(nodeId);
    }

    /// @brief Emit the edges that are stored in the vector,
    /// it must be called after generating the blocks and
    /// statements
    void emitAllEdgeStmts(mlir::Operation *op)
    {
        assert(mlir::isa<mlir::FunctionOpInterface>(op) &&
               "Operation provided must implement FunctionOpInterface.");

        auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op);

        for (auto &region : func->getRegions())
        {
            for (auto &block : region.getBlocks())
            {
                auto terminator_instr = block.getTerminator();
                auto terminator_node = instrToNode.at(terminator_instr);

                if (auto cb = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator_instr))
                {
                    auto &true_block_insns = cb.getTrueDest()->getOperations().front();
                    auto &false_block_insns = cb.getFalseDest()->getOperations().front();

                    os << "v" << terminator_node.id;
                    os << " -> ";
                    os << "v" << instrToNode.at(&true_block_insns).id;
                    os << "[style=\"solid,bold\",color=green,weight=10,constraint=true];\n";

                    os << "v" << terminator_node.id;
                    os << " -> ";
                    os << "v" << instrToNode.at(&false_block_insns).id;
                    os << "[style=\"solid,bold\",color=red,weight=10,constraint=true];\n";
                }
                else if (auto b = mlir::dyn_cast<mlir::cf::BranchOp>(terminator_instr))
                {
                    auto & dest_block_insns = b.getDest()->getOperations().front();

                    os << "v" << terminator_node.id;
                    os << " -> ";
                    os << "v" << instrToNode.at(&dest_block_insns).id;
                    os << "[style=\"solid,bold\",color=blue,weight=10,constraint=true];\n";
                }
                else if (auto ft = mlir::dyn_cast<mlir::KUNAI::MjolnIR::FallthroughOp>(terminator_instr))
                {
                    auto & dest_block_insns = ft.getDest()->getOperations().front();

                    os << "v" << terminator_node.id;
                    os << " -> ";
                    os << "v" << instrToNode.at(&dest_block_insns).id;
                    os << "[style=\"solid,bold\",color=black,weight=10,constraint=true];\n";
                }
            }
        }
    }

    /// Generate a label for an operation.
    std::string getLabel(mlir::Operation *op)
    {
        return strFromOs(
            [&](mlir::raw_ostream &os)
            {
                // Print operation name and type.
                os << op->getName();

                // if (printResultTypes) {
                os << " : (";
                std::string buf;
                llvm::raw_string_ostream ss(buf);
                interleaveComma(op->getResultTypes(), ss);
                os << truncateString(ss.str()) << ")";
                //os << ")";
                //}

                // Print attributes.
                // if (printAttrs) {
                os << "\n";
                for (const mlir::NamedAttribute &attr : op->getAttrs())
                {
                    os << '\n'
                       << attr.getName().getValue() << ": ";
                    emitMlirAttr(os, attr.getValue());
                }
                //}
            });
    }

    /// Truncate long strings.
    std::string truncateString(std::string str)
    {
        if (str.length() <= maxLabelLen)
            return str;
        return str.substr(0, maxLabelLen) + "...";
    }

    /// Output where to write the DOT file
    mlir::raw_indented_ostream os;
    /// vector with the edges, this will be emitted at the end of the graph
    /// generation, it will make printing easier.
    std::vector<std::string> edges;
    /// Mapping of SSA values to Graphviz nodes/clusters.
    mlir::DenseMap<mlir::Operation *, Node> instrToNode;
    /// Counter for generating unique node/subgraph identifiers.
    int counter = 0;
};

namespace KUNAI
{
    namespace MjolnIR
    {
        std::unique_ptr<mlir::Pass> createMjolnIROpGraphPass(mlir::raw_ostream &os)
        {
            return std::make_unique<PrintOpPass>(os);
        }
    }
}