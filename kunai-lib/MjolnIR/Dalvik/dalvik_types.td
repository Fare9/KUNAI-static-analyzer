//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
// @author Ernesto Java <javaernesto@gmail.com>
//
// @file mjolnir_vars.td
// @brief definition of all data types from MjolnIR IR for Dalvik

#ifndef KUNAI_MJOLNIR_MLIR_MJOLNIR_VARS_TD
#define KUNAI_MJOLNIR_MLIR_MJOLNIR_VARS_TD

include "Kunai/MjolnIR_MLIR/mjolnir_base.td"

//===----------------------------------------------------------------------===//
// DVMType
//===----------------------------------------------------------------------===//
class DVMType<string name, string typeMnemonic, list<Trait> traits = [],
                    string baseCppClass = "::KUNAI::MjolnIR::DVMType"> 
                : TypeDef<MjolnIR_Dialect, name, traits, baseCppClass>
{
    let summary = "Abstract type from Dalvik Machine";

    let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Definition of fundamental types
//===----------------------------------------------------------------------===//
class DVMFundamental : DVMType<string name, string typeMnemonic>
    : DVMType<name, typeMnemonic>
{
    let summary = "Base class for the Fundamental types";

    let description = [{
        This Fundamental class will be base for all the other
        types from the Java language used by the Dalvik Virtual
        Machine. We will specify also the way of printing
    }];

    let assemblyFormat = "$name";
}

def DVMVoidType : DVMFundamental<"void", "V"> {
  let summary = "The 'void' type from the Dalvik Virtual Machine";
}

def DVMBooleanType : DVMFundamental<"bool", "Z"> {
  let summary = "The 'boolean' type from the Dalvik Virtual Machine";
}

def DVMByteType : DVMFundamental<"i8", "B"> {
  let summary = "The 'byte' type from the Dalvik Virtual Machine";
}

def DVMCharType : DVMFundamental<"i16", "C"> {
  let summary = "The 'char' type from the Dalvik Virtual Machine";
}

def DVMShortType : DVMFundamental<"i16", "S"> {
  let summary = "The 'short' type from the Dalvik Virtual Machine";
}

def DVMIntType : DVMFundamental<"i32", "I"> {
  let summary = "The 'int' type from the Dalvik Virtual Machine";
}

def DVMLongType : DVMFundamental<"i64", "J"> {
  let summary = "The 'long' type from the Dalvik Virtual Machine";
}

def DVMFloatType : DVMFundamental<"f32", "F"> {
  let summary = "The 'float' type from the Dalvik Virtual Machine";
}

def DVMDoubleType : DVMFundamental<"f64", "D"> {
  let summary = "The 'double' type from the Dalvik Virtual Machine";
}



//===----------------------------------------------------------------------===//
// DVMClass
//===----------------------------------------------------------------------===//
class DVMClass<string className, list<Trait> traits = [], 
                    string baseCppClass = "::KUNAI::MjolnIR::DVMType">
    : DVMType<className, "class", traits, baseCppClass>
{
    let summary = "Class from Dalvik";

    let description = [{
        This class represents a Class from Java compiled to Dalvik
        Virtual Machine, the class just needs a string for the name.
    }];

    let arguments = (ins
        AnyType:$context,
        StringAttr:$className
    );

    let assemblyFormat = "$className";

    let printingHook = [{
        getCustomPrinterFuncName = "printClassName";
    }];
}

//===----------------------------------------------------------------------===//
// DVMArrayType
//===----------------------------------------------------------------------===//
class DVMArrayType<unsigned depth, DVMType elementType,
                    list<Trait> traits = [],
                    string baseCppClass = "::KUNAI::MjolnIR::DVMType"> 
                : DVMType<("[" + str(depth) + " x " + elementType.getDebugTypeName() + "]"), "array", traits, baseCppClass> {
  let summary = "Array type from Dalvik Virtual Machine";

  let description = [{
    Array type from the Dalvik Virtual Machine, we need to specify
    the depth of the array and the element type from the Array.
  }];

  let arguments = (ins AnyType:$context, Type:$elementType);

  let assemblyFormat = "[" + str(depth) + " x $elementType]";

  let printingHook = [{
      getCustomPrinterFuncName = "printArrayType";
  }];
}

#endif