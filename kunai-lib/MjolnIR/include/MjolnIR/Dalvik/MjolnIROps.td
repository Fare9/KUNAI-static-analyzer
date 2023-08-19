//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.td
// @brief Operations from Dalvik instructions

#ifndef DALVIK_MJOLNIROPS_TD
#define DALVIK_MJOLNIROPS_TD

include "MjolnIR/Dalvik/MjolnIRTypes.td"

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
        AnyType:$src
    );

    let results = (outs
        AnyType:$dst
    );
}

class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Binary operation receives two Values of AnyTypes,
        each one of these operations return only one Value.
    }];

    let arguments = (ins
        AnyType:$lhs,
        AnyType:$rhs
    );

    let results = (outs
        AnyType:$res
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
def MjolnIR_LoadValue : MjolnIR_Op<"load-value", [Pure]>
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
        AnyType:$res
    );

    let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Load string operations
//===----------------------------------------------------------------------===//
def MjolnIR_LoadString : MjolnIR_Op<"load-string", [Pure]>
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
        AnyType:$res
    );

    let assemblyFormat = "`(` $string `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// LoadFieldOp
//===----------------------------------------------------------------------===//
def LoadFieldOp : MjolnIR_Op<"loadfield",
    [Pure]>
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
        MjolnIR_FieldType:$access,      // How access to the field is done
        FlatSymbolRefAttr:$fieldName, // name of Field
        FlatSymbolRefAttr:$fieldClass, // class of Field
        UI32Attr: $fieldRef, // reference to Field in Dalvik code
        Optional<AnyType>:$object   // in case it is an instance field, here it goes the object
    );

    let results = (outs
        AnyType               // any of the DVM Basic Types
    );

    let builders = [
        OpBuilder<
            (ins "Type":$resultType, "StringRef":$fieldName, "StringRef":$fieldClass, "uint32_t":$fieldRef), [{
    /** In case of access to a static field */
    odsState.addAttribute(getAccessAttrName(odsState.name), ::mlir::KUNAI::MjolnIR::FieldTypeAttr::get(odsBuilder.getContext(), FieldType::STATIC));
    odsState.addAttribute(getFieldNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldName));
    odsState.addAttribute(getFieldClassAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldClass));
    odsState.addAttribute(getFieldRefAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32, /*isSigned=*/false), fieldRef));
    odsState.addTypes(resultType);
}]
        >,
        OpBuilder<
            (ins "Type":$resultType, "StringRef":$fieldName, "StringRef":$fieldClass, "uint32_t":$fieldRef, "Value":$object), [{
    /** In case of access to a field through an instance of an object */
    odsState.addOperands(object);
    odsState.addAttribute(getAccessAttrName(odsState.name), ::mlir::KUNAI::MjolnIR::FieldTypeAttr::get(odsBuilder.getContext(), FieldType::INSTANCE));
    odsState.addAttribute(getFieldNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldName));
    odsState.addAttribute(getFieldClassAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldClass));
    odsState.addAttribute(getFieldRefAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32, /*isSigned=*/false), fieldRef));
    odsState.addTypes(resultType);
}]
        >
    ];

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
        AnyType:$value,                 // value to store in the Field
        MjolnIR_FieldType:$access,      // How access to the field is done
        FlatSymbolRefAttr:$fieldName,   // name of Field
        FlatSymbolRefAttr:$fieldClass,  // class of Field
        UI32Attr: $fieldRef,            // reference to Field in Dalvik code
        Optional<AnyType>:$object   // in case it is an instance field, here it goes the object
    );

    let builders = [
        OpBuilder<
            (ins "Value":$value, "StringRef":$fieldName, "StringRef":$fieldClass, "uint32_t":$fieldRef), [{
    /** In case of access to a static field */
    odsState.addOperands(value);
    odsState.addAttribute(getAccessAttrName(odsState.name), ::mlir::KUNAI::MjolnIR::FieldTypeAttr::get(odsBuilder.getContext(), FieldType::STATIC));
    odsState.addAttribute(getFieldNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldName));
    odsState.addAttribute(getFieldClassAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldClass));
    odsState.addAttribute(getFieldRefAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32, /*isSigned=*/false), fieldRef));
}]
        >,
        OpBuilder<
            (ins "Value":$value, "StringRef":$fieldName, "StringRef":$fieldClass, "uint32_t":$fieldRef, "Value":$object), [{
    /** In case of access to a field through an instance of an object */
    odsState.addOperands(value);
    odsState.addOperands(object);
    odsState.addAttribute(getAccessAttrName(odsState.name), ::mlir::KUNAI::MjolnIR::FieldTypeAttr::get(odsBuilder.getContext(), FieldType::INSTANCE));
    odsState.addAttribute(getFieldNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldName));
    odsState.addAttribute(getFieldClassAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), fieldClass));
    odsState.addAttribute(getFieldRefAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32, /*isSigned=*/false), fieldRef));
}]
        >
    ];

    

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
        Variadic<AnyType>: $operands // the inputs that must be the same
                                         // of those from the callee
    );

    // The return type must also be the same than the returned type
    // from the callee
    let results = (outs
        Optional<AnyType>
    );

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
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

    let extraClassDeclaration = [{
        FunctionType getCalleeType();


        /// return iterator to the beginning of the operands
        operand_iterator arg_operand_begin()
        {
            return operand_begin();
        }

        /// return iterator to the end of the operands
        operand_iterator arg_operand_end()
        {
            return operand_end();
        }
    }];
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
        Variadic<AnyType>:$input
    );

    // return operation only emits the input in the format if it is present.
    let assemblyFormat = "attr-dict ($input^ `:` type($input))?";

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
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
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

def TryOp : MjolnIR_Op<"try_block">
{
    let summary = "Instruction to specify that the block is a try block";

    let description = [{
        This instruction will specify that the block is a try
        block, when we find this instruction during the transformation
        from MLIR to Dalvik bytecode, we set the block as a
        try block.
    }];

    let assemblyFormat = [{
        `:` attr-dict
    }];
}

def CatchOp : MjolnIR_Op<"catch_block">
{
    let summary = "Instruction to specify that the block is a catch block";

    let description = [{
        This instruction will specify that the block is a catch
        block, when we find this instruction during the transformatino
        from MLIR to Dalvik bytecode, we set the block as a
        catch block.
    }];

    let assemblyFormat = [{
        `:` attr-dict
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
        AnyType:$size
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
        AnyType:$value
    );
}

//===----------------------------------------------------------------------===//
// UndefOp from Numba project to create a value to use in transformations
//===----------------------------------------------------------------------===//
def UndefOp : MjolnIR_Op<"undef", [Pure]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasCanonicalizer = 1;
}


#endif // DALVIK_MJOLNIROPS_TD