//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRTypes.td
// @brief Definition of types from the Dalvik Virtual Machine.

#ifndef DALVIK_MJOLNIRTYPES_TD
#define DALVIK_MJOLNIRTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dalvik/MjolnIRDialect.td"

class MjolnIRType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<MjolnIR_Dialect, typeName, traits>
{
  let summary = typeName;
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Basic Types
//===----------------------------------------------------------------------===//

def DVMVoid : MjolnIRType<"DVMVoid", "dvmvoid"> {
  let summary = "Void type from Dalvik, returns nothing";
}

def DVMByte : MjolnIRType<"DVMByte", "dvmbyte"> {
  let summary = "Bool type from Dalvik";
}

def DVMBool : MjolnIRType<"DVMBool", "dvmbool"> {
  let summary = "Bool type from Dalvik";
}

def DVMChar : MjolnIRType<"DVMChar", "dvmchar"> {
  let summary = "Char type from Dalvik";
}

def DVMShort : MjolnIRType<"DVMShort", "dvmshort"> {
  let summary = "Short type from Dalvik";
}

def DVMInt : MjolnIRType<"DVMInt", "dvmint"> {
  let summary = "Integer type from Dalvik";
}

def DVMLong : MjolnIRType<"DVMLong", "dvmlong"> {
  let summary = "Long type from Dalvik";
}

def DVMFloat : MjolnIRType<"DVMFloat", "dvmfloat"> {
  let summary = "Float type from Dalvik";
}

def DVMDouble : MjolnIRType<"DVMDouble", "dvmdouble"> {
  let summary = "Double type from Dalvik";
}

def DVMObject : MjolnIRType<"DVMObject", "object"> {
  let summary = "An object from any Java class";

  let parameters = (ins
    StringRefParameter<"the name of the class">:$value
  );

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

def DVMArrayType : MjolnIRType<"DVMArray", "array">
{
  let summary = "Simple MjolnIR Array Type";

  let description = [{
    This type represents a fixed-size array from Dalvik. It contains
    the size of the array and the type of the array.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];


  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    /// Checks the type inside of the array.
    static bool isValidElementType(Type type);
  }];
}

def DVM_BasicType : AnyTypeOf<[DVMBool, DVMChar, DVMShort, DVMInt, DVMLong, DVMFloat, DVMDouble]>;


#endif // DALVIK_MJOLNIRTYPES_TD