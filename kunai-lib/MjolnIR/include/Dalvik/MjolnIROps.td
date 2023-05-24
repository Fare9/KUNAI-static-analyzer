//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.td
// @brief Operations from Dalvik instructions

#ifndef DALVIK_MJOLNIROPS_TD
#define DALVIK_MJOLNIROPS_TD

include "Dalvik/MjolnIRTypes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"



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

class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Binary operation receives two Values of DVM_BasicTypes,
        each one of these operations return only one Value.
    }];

    let arguments = (ins
        DVM_BasicType:$lhs,
        DVM_BasicType:$rhs
    );

    let results = (outs
        DVM_BasicType:$res
    );
}

class MjolnIR_ComparisonOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Different type of comparisons in Dalvik, the types of the arguments
        can be any of the Dalvik Machine, but always as result the operation
        returns a bool value for ControlFlowOps Dialect (I1 value)
    }];

    let arguments = (ins
        DVM_BasicType:$lhs,
        DVM_BasicType:$rhs
    );

    let results = (outs
        I1:$res
    );
}

def MjolnIR_Nop : MjolnIR_Op<"nop", [Pure]>
{
    let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//
def MjolnIR_MoveOp : MjolnIR_UnaryOpBase<"move">;
def MjolnIR_CastOp : MjolnIR_UnaryOpBase<"cast">;

def MjolnIR_Neg : MjolnIR_UnaryOpBase<"neg", [AllTypesMatch<["src", "dst"]>]>;
def MjolnIR_Not : MjolnIR_UnaryOpBase<"not", [AllTypesMatch<["src", "dst"]>]>;

//===----------------------------------------------------------------------===//
// Load values operations
//===----------------------------------------------------------------------===//
def MjolnIR_LoadValue : MjolnIR_Op<"load-value", [Pure, ConstantLike]>
{
    let summary = "Load a constant value";

    let description = [{
        In order to generate a SSA value from a constant one we have to create
        an instruction that has a value as attribute and returns a Value to any
        of the types from Dalvik, in this way the generated value is already in
        SSA value.

        Examples:

        ```mlir
        // Example of data loaded
        %0 = MjolnIR.load-value(33 : dvmint) : dvmint
        ```
    }];

    let arguments = (ins
        SI64Attr:$value
    );

    let results = (outs
        DVM_BasicType:$res
    );

    let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Load string operations
//===----------------------------------------------------------------------===//
def MjolnIR_LoadString : MjolnIR_Op<"load-string", [Pure, ConstantLike]>
{
    let summary = "Load a String into a String object";

    let description = [{
        In order to generate a SSA value from a loaded String, we have to create
        an instruction which attribute will be the loaded string, and the return
        will be a DVMObject with the attribute 'java/lang/String'.

        Examples:

        ```mlir
        %0 = MjolnIR.load-string("ExampleString" : string) : dvmobject
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$string,
        UI32Attr:$stringRef
    );

    let results = (outs
        DVM_BasicType:$res
    );

    let assemblyFormat = "`(` $string `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//
def MjolnIR_AddOp : MjolnIR_BinaryOpBase<"add", [Commutative]>
{
    let summary = "Add two DVM registers";
}

def MjolnIR_SubOp : MjolnIR_BinaryOpBase<"sub">
{
    let summary = "Substract two DVM registers";
}

def MjolnIR_MulOp : MjolnIR_BinaryOpBase<"mul", [Commutative]>
{
    let summary = "Multiply two DVM registers";
}

def MjolnIR_DivOp : MjolnIR_BinaryOpBase<"div">
{
    let summary = "Divide two DVM registers";
}

def MjolnIR_RemOp : MjolnIR_BinaryOpBase<"rem">
{
    let summary = "Divide two DVM registers and get the remainder";
}

def MjolnIR_AndOp : MjolnIR_BinaryOpBase<"and">
{
    let summary = "Apply AND operation in two DVM registers";
}

def MjolnIR_OrOp  : MjolnIR_BinaryOpBase<"or">
{
    let summary = "Apply OR operation in two DVM registers";
}

def MjolnIR_XorOp : MjolnIR_BinaryOpBase<"xor">
{
    let summary = "Apply XOR operation in two DVM registers";
}

def MjolnIR_Shl   : MjolnIR_BinaryOpBase<"shl">
{
    let summary = "Shift Left a DVM register";
}

def MjolnIR_Shr   : MjolnIR_BinaryOpBase<"shr">
{
    let summary = "Shift Right a DVM register";
}

def MjolnIR_UShr  : MjolnIR_BinaryOpBase<"ushr">
{
    let summary = "Unsigned Shift Right a DVM register";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def MjolnIR_CmpEq : MjolnIR_ComparisonOpBase<"cmp-eq", [Commutative]>
{
    let summary = "Compare if values are equal";
}

def MjolnIR_CmpNEq : MjolnIR_ComparisonOpBase<"cmp-neq", [Commutative]>
{
    let summary = "Compare if values are not equal";
}

def MjolnIR_CmpLt : MjolnIR_ComparisonOpBase<"cmp-lt">
{
    let summary = "Compare a value is lower than other";
}

def MjolnIR_CmpGe : MjolnIR_ComparisonOpBase<"cmp-ge">
{
    let summary = "Compare a value is greater or equal than other";
}

def MjolnIR_CmpGt : MjolnIR_ComparisonOpBase<"cmp-gt">
{
    let summary = "Compare a value is greater than other";
}

def MjolnIR_CmpLe : MjolnIR_ComparisonOpBase<"cmp-le">
{
    let summary = "Compare a value is lower or equal than other";
}

def MjolnIR_CmpEqz : MjolnIR_ComparisonOpBase<"cmp-eqz">
{
    let summary = "Compare a value is equal to zero";
}

def MjolnIR_CmpNeqz : MjolnIR_ComparisonOpBase<"cmp-neqz">
{
    let summary = "Compare a value is not equal to zero";
}

def MjolnIR_CmpLtz : MjolnIR_ComparisonOpBase<"cmp-ltz">
{
    let summary = "Compare a value is lower than zero";
}

def MjolnIR_CmpGez : MjolnIR_ComparisonOpBase<"cmp-gez">
{
    let summary = "Compare a value is greater or equal to zero";
}

def MjolnIR_CmpGtz : MjolnIR_ComparisonOpBase<"cmp-gtz">
{
    let summary = "Compare a value is greater than zero";
}

def MjolnIR_CmpLez : MjolnIR_ComparisonOpBase<"cmp-lez">
{
    let summary = "Compare a value is lower or equal to zero";
}



//===----------------------------------------------------------------------===//
// LoadFieldOp
//===----------------------------------------------------------------------===//
def LoadFieldOp : MjolnIR_Op<"loadfield",
    [ConstantLike, Pure]>
{
    let summary = "Load a field value";

    let description = [{
        In order to produces an SSA value from a symbol reference
        we have this operation that takes a symbol field name and
        returns a value with the type of the field.

        ```mlir
            %2 = MjolnIR.loadfield @MyField (3) :  () -> DVMType
        ```

        It is not possible to create a kind of generic variable so
        this will be a way to obtain a value from a non-SSA type
        since the nature of a Field is not being SSA.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$fieldName, // name of Field
        FlatSymbolRefAttr:$fieldClass, // class of Field
        UI32Attr: $fieldRef // reference to Field in Dalvik code
    );

    let results = (outs
        DVM_BasicType               // any of the DVM Basic Types
    );

    let assemblyFormat = "attr-dict $fieldClass`->`$fieldName `(` $fieldRef `)` `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StoreFieldOp
//===----------------------------------------------------------------------===//
def StoreFieldOp : MjolnIR_Op<"storefield", []>
{
    let summary = "Store value in field";

    let description = [{
        In order to store a value in a field we have to represent the field
        as a Symbol, since there are not global variables. In that case we
        will obtain the SSA value that will be stored in the field, and we will
        obtain the name of the field and reference.

        ```mlir
            MjolnIR.storeField 
        ```
    }];

    let arguments = (ins
        DVM_BasicType:$value,   // value to store in the Field
        FlatSymbolRefAttr:$fieldName, // name of Field
        FlatSymbolRefAttr:$fieldClass, // class of Field
        UI32Attr: $fieldRef // reference to Field in Dalvik code
    );

    let assemblyFormat = "attr-dict $value`:`type($value)`,` $fieldClass`->`$fieldName `(`$fieldRef`)`";
}


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
        UI32Attr:$methodRef,
        UI32Attr:$isStatic,
        Variadic<DVM_BasicType>: $inputs // the inputs that must be the same
                                         // of those from the callee
    );

    // The return type must also be the same than the returned type
    // from the callee
    let results = (outs
        Optional<DVM_BasicType>
    );

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments, "MethodOp& ":$method)>,
        OpBuilder<(ins "MethodOp":$callee, "IntegerAttr":$methodRef, "IntegerAttr":$isStatic, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addAttribute("methodRef", methodRef);
            $_state.addAttribute("isStatic", isStatic);
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "Type":$result, "StringAttr":$callee, "IntegerAttr":$methodRef, "IntegerAttr":$isStatic, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addAttribute("methodRef", methodRef);
            $_state.addAttribute("isStatic", isStatic);
            if (!mlir::isa<::mlir::KUNAI::MjolnIR::DVMVoidType>(result))
                $_state.addTypes(result);
        }]>,
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

//===----------------------------------------------------------------------===//
// FallthroughOp
//===----------------------------------------------------------------------===//

def FallthroughOp : MjolnIR_Op<"fallthrough", [
    Pure, Terminator
  ]> {
  let summary = "FallThrough jump";
  let description = [{
    Fallthrough instruction is used as a fake jump for implementing the logic
    of control flow graph, two blocks must be connected by a terminator
    instruction, and a fallthrough in the dalvik graph is not any kind of jump.
    For this we need to add a fallthrough as an unconditional jump.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      MjolnIR.fallthrough ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                   CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// New operation
//===----------------------------------------------------------------------===//
def NewOp : MjolnIR_Op<"new", [Pure]>
{
    let summary = "Create a new object";

    let description = [{
        This instruction creates a SSA Value for an object
        the object will have as one of its parameter the
        name of the class.
    }];

    let results = (outs
        AnyType // object created with new
    );
}

//===----------------------------------------------------------------------===//
// NewArray operation
//===----------------------------------------------------------------------===//
def NewArrayOp : MjolnIR_Op<"new-array", [Pure]>
{
    let summary = "Create a new array object";

    let description = [{
        This instruction creates a SSA Value for an array
        type, the array will contain the base type and the
        size of the array.
    }];

    let arguments = (ins
        DVM_BasicType:$size
    );

    let results = (outs
        AnyType
    );

}


//===----------------------------------------------------------------------===//
// LoadValueArray operation
//===----------------------------------------------------------------------===//
def LoadValueArrayOp : MjolnIR_Op<"load-value-array", [Pure]>
{
    let summary = "Create an access to an array with an index";

    let description = [{
        This instruction makes an access to an index from an array
        for writing.
    }];

    let arguments = (ins
        AnyType:$base,
        UI32Attr:$index,
        DVM_BasicType:$value
    );
}


#endif // DALVIK_MJOLNIROPS_TD