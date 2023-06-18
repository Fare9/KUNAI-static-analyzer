//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRTypes.td
// @brief Definition of types from the Dalvik Virtual Machine.

#ifndef DALVIK_MJOLNIRTYPES_TD
#define DALVIK_MJOLNIRTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "MjolnIR/Dalvik/MjolnIRDialect.td"

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

#endif // DALVIK_MJOLNIRTYPES_TD