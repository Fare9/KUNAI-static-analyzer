# KUNAI-static-analyzer

Tool aimed to provide a binary analysis of different file formats through the use of an Intermmediate Representation.

## MjolnIR

IR written for generic analysis of different file formats and binary architectures, this IR allows us to represent binaries in a higher language, then create Control-Flow graphs, Data-Flow graphs, etc. MjolnIR is intended to support by default three-address code and Abstract-Syntax Tree (AST).

### IRStmnt

At the top of the IR we have the statements, these will be every instruction from the IR that could be executed by the program, between these IRStmnt are the expressions (explained later), but more specific statements are those that change the Control-Flow Graph from the function/method, these are conditional and unconditional jumps, return statements, etc.

```
IRStmnt     -->     IRUJmp   |
                    IRCJmp   |
                    IRRet    |
                    IRBlock  |
                    IRNop    |
                    IRSwitch |
                    IRExpr 

IRUJmp      -->     jmp addr
IRCJmp      -->     if (IRStmnt) { jmp addr } NEXT fallthrough_addr
IRRet       -->     Ret IRStmnt
IRBlock     -->     IRStmnt1, IRStmnt2, ..., IRStmntN
```

### IRExpr

The IR involves to support various instructions from the code, these are what we call IRExpr, these kind of instructions don't modify the control flow of the code but apply different kind of operations to the variables/registers/memory in the program.

```
IRExpr    -->   IRBinOp   |
                IRUnaryOp | 
                IRAssign  |
                IRCall    |
                IRLoad    |
                IRStore   |
                IRZComp   |
                IRBComp   |
                IRNew     |
                IRType    
                
IRBinOp   -->   IRExpr <- IRExpr bin_op_t IRExpr
IRUnaryOp -->   IRExpr <- unary_op_t IRExpr
IRAssign  -->   IRExpr <- IRExpr
IRCall    -->   IRExpr(IRExpr1, IRExpr2, ..., IRExprN)
IRLoad    -->   IRExpr <- *IRExpr
IRStore   -->   *IRExpr <- IRExpr
IRZComp   -->   IRExpr zero_comp_t
IRBComp   -->   IRExpr comp_t IRExpr
IRNew     -->   IRExpr <- new IRExpr

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
zero_comp_t  -->   EQUAL_ZERO_T |
                   NOT_EQUAL_ZERO_T
comp_t       -->   EQUAL_T              |
                   NOT_EQUAL_T          |
                   GREATER_T            |
                   GREATER_EQUAL_T      |
                   LOWER_T              |
                   ABOVE_T              |
                   ABOVE_EQUAL_T        |
                   BELOW_T                           
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
               IRClass  |
               NONE
```

### Dalvik

Dalvik has been the first "assembly" language to be supported with **MjolnIR** inside of **KUNAI**, you can check all the opcodes and its translation in [Dalvik Opcodes](./doc/MJOLNIR/dalvik_opcodes.md)

### References

Here there are some references used as ideas and base for the design and/or implementation of **KUNAI** and **MjolnIR**:

* Projects:
    * [Androguard](https://github.com/androguard/androguard)
    * [Miasm](https://github.com/cea-sec/miasm/tree/master)
    * [VEX](https://github.com/angr/vex)
* Papers:
    * [Wire -- A Formal Intermediate Language for Binary Analysis](https://ieeexplore.ieee.org/document/6296015)
    * [REIL: A platform-independent intermediate representation of disassembled code for static code analysis](https://static.googleusercontent.com/media/www.zynamics.com/en//downloads/csw09.pdf)
    * [Dexpler: Converting Android Dalvik Bytecode to Jimple for Static Analysis with Soot](https://arxiv.org/pdf/1205.3576.pdf)
* Books:
    * [Advanced Compiler Design and Implementation](https://www.amazon.com/-/es/Steven-Muchnick/dp/1558603204)
    * [An Introduction to the Theory of Optimizing Compilers](https://www.amazon.com/-/es/Jonas-Skeppstedt/dp/172593048X)
    * [Introduction to Compilers and Language Design: Second Edition](https://www3.nd.edu/~dthain/compilerbook/compilerbook.pdf)
