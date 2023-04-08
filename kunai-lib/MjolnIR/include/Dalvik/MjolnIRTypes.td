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

class MjolnIRBuiltIntType<string typeName, list<Trait> traits = []>
  : TypeDef<MjolnIR_Dialect, typeName, traits>
{
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// Basic Types
//===----------------------------------------------------------------------===//

def DVMVoid : MjolnIRBuiltIntType<"DVMVoid"> {
  let summary = "Void type from Dalvik, returns nothing";
}

def DVMByte : MjolnIRBuiltIntType<"DVMByte"> {
  let summary = "Bool type from Dalvik";
}

def DVMBool : MjolnIRBuiltIntType<"DVMBool"> {
  let summary = "Bool type from Dalvik";
}

def DVMChar : MjolnIRBuiltIntType<"DVMChar"> {
  let summary = "Char type from Dalvik";
}

def DVMShort : MjolnIRBuiltIntType<"DVMShort"> {
  let summary = "Short type from Dalvik";
}

def DVMInt : MjolnIRBuiltIntType<"DVMInt"> {
  let summary = "Integer type from Dalvik";
}

def DVMLong : MjolnIRBuiltIntType<"DVMLong"> {
  let summary = "Long type from Dalvik";
}

def DVMFloat : MjolnIRBuiltIntType<"DVMFloat"> {
  let summary = "Float type from Dalvik";
}

def DVMDouble : MjolnIRBuiltIntType<"DVMDouble"> {
  let summary = "Double type from Dalvik";
}

def DVM_BasicType : AnyTypeOf<[DVMBool, DVMChar, DVMShort, DVMInt, DVMLong, DVMFloat, DVMDouble]>;


#endif // DALVIK_MJOLNIRTYPES_TD