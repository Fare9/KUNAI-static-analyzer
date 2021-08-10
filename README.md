# KUNAI-static-analyzer

Tool aimed to provide a binary analysis of different file formats through the use of an Intermmediate Representation.

## MjolnIR

IR written for generic analysis of different file formats and binary architectures, this IR allows us to represent binaries in a higher language, then create Control-Flow graphs, Data-Flow graphs, etc. MjolnIR is intended to support by default three-address code and Abstract-Syntax Tree (AST).

### IRExpr

The IR involves to support various instructions from the code, these are what we call IRExpr, these kind of instructions don't modify the control flow of the code but apply different kind of operations to the variables/registers/memory in the program.

```
IRExpr    -->   IRBinOp   |
                IRUnaryOp | 
                IRAssign  |
                IRCall    |
                IRType    
                
IRBinOp   -->   IRExpr <- IRExpr bin_op_t IRExpr
IRUnaryOp -->   IRExpr <- unary_op_t IRExpr
IRAssign  -->   IRExpr <- IRExpr
IRCall    -->   IRExpr(IRExpr1, IRExpr2, ..., IRExprN)

# kind of operations
bin_op_t  -->   ADD_OP_T   |
                SUB_OP_T   |
                S_MUL_OP_T |
                U_MUL_OP_T |
                S_DIV_OP_T |
                U_DIV_OP_T |
                MOD_OP_T
unary_op_t   -->   INC_OP_T    |
                   DEC_OP_T    |
                   NOT_OP_T    |
                   NEG_OP_T    |
                   CAST_OP_T   |
                   Z_EXT_OP_T  |
                   S_EXT_OP_T
```

### IRType

For supporting the types we find in the binary code, we have written a serie of classes which derive from a super class named **IRType**, as derived classes we have: *registers*, *constant values*, *strings*, *memory*, *callee types*.

```
IRType   -->   IRReg |
               IRTempReg |
               IRConstInt |
               IRMemory |
               IRString |
               IRCallee |
               NONE
```