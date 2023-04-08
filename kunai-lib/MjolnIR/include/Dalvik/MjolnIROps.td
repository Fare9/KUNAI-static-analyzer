//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.td
// @brief Operations from Dalvik instructions

#ifndef DALVIK_MJOLNIROPS_TD
#define DALVIK_MJOLNIROPS_TD

include "Dalvik/MjolnIRTypes.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"




/// Base class for unary operations, we specify just one
/// argument value as entry, and one value as result
class MjolnIR_UnaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let arguments = (ins
        DVM_BasicType:$src
    );

    let results = (outs
        DVM_BasicType:$dst
    );
}

class MjolnIR_LoadNumber<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let arguments = (ins
        AnyInteger:$src
    );

    let results = (outs
        DVM_BasicType:$dst
    );
}

class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let arguments = (ins
        DVM_BasicType:$lhs,
        DVM_BasicType:$rhs
    );

    let results = (outs
        DVM_BasicType:$res
    );
}

// define unary operations
def MjolnIR_MoveOp : MjolnIR_UnaryOpBase<"move">;
def MjolnIR_CastOp : MjolnIR_UnaryOpBase<"cast">;

// define move operation in some way...
def MjolnIR_LoadByte : MjolnIR_LoadNumber<"loadB">;
def MjolnIR_LoadShort : MjolnIR_LoadNumber<"loadS">;
def MjolnIR_LoadInt : MjolnIR_LoadNumber<"loadI">;
def MjolnIR_LoadLong : MjolnIR_LoadNumber<"loadL">;

// define binary operations
def MjolnIR_AddOp : MjolnIR_BinaryOpBase<"add", [Commutative]>;
def MjolnIR_SubOp : MjolnIR_BinaryOpBase<"sub">;
def MjolnIR_MulOp : MjolnIR_BinaryOpBase<"mul", [Commutative]>;
def MjolnIR_DivOp : MjolnIR_BinaryOpBase<"div">;
def MjolnIR_RemOp : MjolnIR_BinaryOpBase<"rem">;
def MjolnIR_AndOp : MjolnIR_BinaryOpBase<"and">;
def MjolnIR_OrOp  : MjolnIR_BinaryOpBase<"or">;
def MjolnIR_XorOp : MjolnIR_BinaryOpBase<"xor">;
def MjolnIR_Shl   : MjolnIR_BinaryOpBase<"shl">;
def MjolnIR_Shr   : MjolnIR_BinaryOpBase<"shr">;
def MjolnIR_UShr  : MjolnIR_BinaryOpBase<"ushr">;

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//
def MethodOp : MjolnIR_Op<"method", [
    DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface,
    IsolatedFromAbove
    ]>
{
    let summary = "Method from Dalvik";

    let description = [{
        A mjolnir.method describes a method from a class from Dalvik, defined
        by the programmer. These will contain instructions from MjolnIR.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];
    
    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];

    let hasCustomAssemblyFormat = 1;
    
    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//
def InvokeOp : MjolnIR_Op<"invoke", [
    DeclareOpInterfaceMethods<CallOpInterface>
]>
{
    let summary = "Invoke operation from Dalvik";

    let description = [{
        Dalvik machine has different invoke-* operations we will use just one,
        and the few corner cases will be transformed to special operations in that
        case.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$callee, // called method
        Variadic<DVM_BasicType>: $inputs // the inputs that must be the same
                                         // of those from the callee
    );

    // The return type must also be the same than the returned type
    // from the callee
    let results = (outs
        Variadic<DVM_BasicType>
    );

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments, "MethodOp& ":$method)>
    ];
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
def ReturnOp : MjolnIR_Op<"return", [
    Pure,
    HasParent<"MethodOp">,
    Terminator
]>
{
    let summary = "return operation from Dalvik";

    let description = [{
        "return" operation represents the return from a function.
        An optional operation is obtained, but no result is produced (no
        other instruction obtains it, since an invoke instruction already
        retrieves an optional value).
    }];

    // an optional value is given as an input to return. This return value
    // must match the return type of the enclosing function.
    let arguments = (ins
    Variadic<DVM_BasicType>:$input
    );

    // return operation only emits the input in the format if it is present.
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    // Allow building a ReturnOp with no return operand.
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];
}

#endif // DALVIK_MJOLNIROPS_TD