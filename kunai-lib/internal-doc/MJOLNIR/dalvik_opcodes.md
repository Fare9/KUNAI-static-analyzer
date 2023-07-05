| Instruction Type | Opcode | Opcode name | IR code |
|:----------------:|:------:|:------------|:--------|
| Instruction10t | 40 | OP_GOTO | cf::BranchOp |
| Instruction10x | 14 | OP_RETURN_VOID | func::ReturnOp |
| | 0 | OP_NOP | KUNAI::MjolnIR::Nop |
| Instruction11n | 18 | OP_CONST_4 | KUNAI::MjolnIR::LoadValue |
| Instruction11x | 15 | OP_RETURN | func::ReturnOp |
| | 16 | OP_RETURN_WIDE | func::ReturnOp |
| | 17 | OP_RETURN_OBJECT | func::ReturnOp |
| | 10 | OP_MOVE_RESULT | Generate result in KUNAI::MjolnIR::InvokeOp |
| | 11 | OP_MOVE_RESULT_WIDE | Generate result in KUNAI::MjolnIR::InvokeOp |
| | 12 | OP_MOVE_RESULT_OBJECT | Generate result in KUNAI::MjolnIR::InvokeOp |
| Instruction12x | 1 | OP_MOVE | KUNAI::MjolnIR::MoveOp |
| | 4 | OP_MOVE_WIDE | KUNAI::MjolnIR::MoveOp |
| | 7 | OP_MOVE_OBJECT | KUNAI::MjolnIR::MoveOp |
| | 176 | OP_ADD_INT_2ADDR | arith::AddIOp |
| | 187 | OP_ADD_LONG_2ADDR | arith::AddIOp |
| | 198 | OP_ADD_FLOAT_2ADDR | arith::AddIOp |
| | 203 | OP_ADD_DOUBLE_2ADDR | arith::AddIOp |
| | 177 | OP_SUB_INT_2ADDR | arith::SubIOp |
| | 188 | OP_SUB_LONG_2ADDR | arith::SubIOp |
| | 199 | OP_SUB_FLOAT_2ADDR | arith::SubIOp |
| | 204 | OP_SUB_DOUBLE_2ADDR | arith::SubIOp |
| | 178 | OP_MUL_INT_2ADDR | arith::MulIOp |
| | 189 | OP_MUL_LONG_2ADDR | arith::MulIOp |
| | 200 | OP_MUL_FLOAT_2ADDR | arith::MulIOp |
| | 205 | OP_MUL_DOUBLE_2ADDR | arith::MulIOp |
| | 179 | OP_DIV_INT_2ADDR | arith::DivSIOp |
| | 190 | OP_DIV_LONG_2ADDR | arith::DivSIOp |
| | 201 | OP_DIV_FLOAT_2ADDR | arith::DivFOp |
| | 206 | OP_DIV_DOUBLE_2ADDR | arith::DivFOp |
| | 180 | OP_REM_INT_2ADDR | arith::RemSIOp |
| | 191 | OP_REM_LONG_2ADDR | arith::RemSIOp |
| | 202 | OP_REM_FLOAT_2ADDR | arith::RemFOp |
| | 207 | OP_REM_DOUBLE_2ADDR | arith::RemFOp |
| | 181 | OP_AND_INT_2ADDR | arith::AndIOp |
| | 192 | OP_AND_LONG_2ADDR | arith::AndIOp |
| | 182 | OP_OR_INT_2ADDR | arith::OrIOp |
| | 193 | OP_OR_LONG_2ADDR | arith::OrIOp |
| | 183 | OP_XOR_INT_2ADDR | arith::XOrIOp |
| | 194 | OP_XOR_LONG_2ADDR | arith::XOrIOp |
| | 184 | OP_SHL_INT_2ADDR | arith::ShLIOp |
| | 195 | OP_SHR_LONG_2ADDR | arith::ShLIOp |
| | 185 | OP_SHR_INT_2ADDR | arith::ShRSIOp |
| | 196 | OP_SHR_LONG_2ADDR | arith::ShRSIOp |
| | 186 | OP_USHR_INT_2ADDR | arith::ShRUIOp |
| | 197 | OP_USHR_LONG_2ADDR |  arith::ShRUIOp |
| | 123 | OP_NEG_INT | KUNAI::MjolnIR::Neg |
| | 125 | OP_NEG_LONG | KUNAI::MjolnIR::Neg |
| | 127 | OP_NEG_FLOAT | KUNAI::MjolnIR::Neg |
| | 128 | OP_NEG_DOUBLE | KUNAI::MjolnIR::Neg |
| | 124 | OP_NOT_INT | KUNAI::MjolnIR::Not |
| | 126 | OP_NOT_LONG | KUNAI::MjolnIR::Not | 
| | 129 | OP_INT_TO_LONG | MjolnIR::CastOp |
| | 136 | OP_FLOAT_TO_LONG | MjolnIR::CastOp |
| | 139 | OP_DOUBLE_TO_LONG | MjolnIR::CastOp |
| | 130 | OP_INT_TO_FLOAT | MjolnIR::CastOp |
| | 133 | OP_LONG_TO_FLOAT | MjolnIR::CastOp |
| | 140 | OP_DOUBLE_TO_FLOAT | MjolnIR::CastOp |
| | 131 | OP_INT_TO_DOUBLE | MjolnIR::CastOp |
| | 134 | OP_LONG_TO_DOUBLE | MjolnIR::CastOp |
| | 137 | OP_FLOAT_TO_DOBULE | MjolnIR::CastOp |
| | 132 | OP_LONG_TO_INT | MjolnIR::CastOp |
| | 135 | OP_FLOAT_TO_INT | MjolnIR::CastOp |
| | 138 | OP_DOUBLE_TO_INT | MjolnIR::CastOp |
| | 141 | OP_INT_TO_BYTE | MjolnIR::CastOp |
| | 142 | OP_INT_TO_CHAR | MjolnIR::CastOp |
| | 143 | OP_INT_TO_SHORT | MjolnIR::CastOp |
| Instruction20t | 41 | OP_GOTO_16 | cf::BranchOp |
| Instruction21c | 34 | OP_NEW_INSTANCE | KUNAI::MjolnIR::NewOp |
| | 26 | OP_CONST_STRING | KUNAI::MjolnIR::LoadString |
| | 96 | OP_SGET | KUNAI::MjolnIR::LoadFieldOp |
| | 97 | OP_SGET_WIDE | KUNAI::MjolnIR::LoadFieldOp |
| | 98 | OP_SGET_OBJECT | KUNAI::MjolnIR::LoadFieldOp |
| | 99 | OP_SGET_BOOLEAN | KUNAI::MjolnIR::LoadFieldOp |
| | 100 | OP_SGET_BYTE | KUNAI::MjolnIR::LoadFieldOp |
| | 101 | OP_SGET_CHAR | KUNAI::MjolnIR::LoadFieldOp |
| | 102 | OP_SGET_SHORT | KUNAI::MjolnIR::LoadFieldOp |
| Instruction21h | 21 | OP_CONST_HIGH16 | arith::ConstantFloatOp |
| | 25 | OP_CONST_WIDE_HIGH16 | arith::ConstantIntOp |
| Instruction21s | 19 | OP_CONST_16 | arith::ConstantIntOp |
| | 22 | OP_CONST_WIDE_16 | arith::ConstantIntOp |
| Instruction21t | 56 | OP_IF_EQZ | arith::CmpIOp + cf::CondBranchOp |
| | 57 | OP_IF_NEZ | arith::CmpIOp + cf::CondBranchOp |
| | 58 | OP_IF_LTZ | arith::CmpIOp + cf::CondBranchOp |
| | 59 | OP_IF_GEZ | arith::CmpIOp + cf::CondBranchOp |
| | 60 | OP_IF_GTZ | arith::CmpIOp + cf::CondBranchOp |
| | 61 | OP_IF_LEZ | arith::CmpIOp + cf::CondBranchOp |
| Instruction22b | 216 | OP_ADD_INT_LIT8 | arith::ConstantIntOp + arith::AddIOp |
| | 217 | OP_SUB_INT_LIT8 | arith::ConstantIntOp + arith::SubIOp |
| | 218 | OP_MUL_INT_LIT8 | arith::ConstantIntOp + arith::MulIOp |
| | 219 | OP_DIV_INT_LIT8 | arith::ConstantIntOp + arith::DivSIOp |
| | 220 | OP_REM_INT_LIT8 | arith::ConstantIntOp + arith::RemSIOp |
| | 221 | OP_AND_INT_LIT8 | arith::ConstantIntOp + arith::AndIOp |
| | 222 | OP_OR_INT_LIT8 | arith::ConstantIntOp + arith::OrIOp |
| | 223 | OP_XOR_INT_LIT8 | arith::ConstantIntOp + arith::XOrIOp |
| | 224 | OP_SHL_INT_LIT8 | arith::ConstantIntOp + arith::ShLIOp |
| | 225 | OP_SHR_INT_LIT8 | arith::ConstantIntOp + arith::ShRSIOp |
| | 226 | OP_USHR_INT_LIT8 | arith::ConstantIntOp + arith::ShRUIOp |
| Instruction22c | 82 | OP_IGET | KUNAI::MjolnIR::LoadFieldOp |
| | 83 | OP_IGET_WIDE | KUNAI::MjolnIR::LoadFieldOp |
| | 84 | OP_IGET_OBJECT | KUNAI::MjolnIR::LoadFieldOp |
| | 85 | OP_IGET_BOOLEAN | KUNAI::MjolnIR::LoadFieldOp |
| | 86 | OP_IGET_BYTE | KUNAI::MjolnIR::LoadFieldOp |
| | 87 | OP_IGET_CHAR | KUNAI::MjolnIR::LoadFieldOp |
| | 88 | OP_IGET_SHORT | KUNAI::MjolnIR::LoadFieldOp |
| | 89 | OP_IPUT | KUNAI::MjolnIR::StoreFieldOp |
| | 90 | OP_IPUT_WIDE | KUNAI::MjolnIR::StoreFieldOp |
| | 91 | OP_IPUT_OBJECT | KUNAI::MjolnIR::StoreFieldOp |
| | 92 | OP_IPUT_BOOLEAN | KUNAI::MjolnIR::StoreFieldOp |
| | 93 | OP_IPUT_BYTE | KUNAI::MjolnIR::StoreFieldOp |
| | 94 | OP_IPUT_CHAR | KUNAI::MjolnIR::StoreFieldOp |
| | 95 | OP_IPUT_SHORT | KUNAI::MjolnIR::StoreFieldOp |
| | 35 | OP_NEW_ARRAY | KUNAI::MjolnIR::NewArrayOp |
| Instruction22s | 208 | OP_ADD_INT_LIT16 | arith::ConstantIntOp + arith::AddIOp |
| | 209 | OP_SUB_INT_LIT16 | arith::ConstantIntOp + arith::SubIOp |
| | 210 | OP_MUL_INT_LIT16 | arith::ConstantIntOp + arith::MulIOp |
| | 211 | OP_DIV_INT_LIT16 | arith::ConstantIntOp + arith::DivSIOp |
| | 212 | OP_REM_INT_LIT16 | arith::ConstantIntOp + arith::RemSIOp |
| | 213 | OP_AND_INT_LIT16 | arith::ConstantIntOp + arith::AndIOp |
| | 214 | OP_OR_INT_LIT16 | arith::ConstantIntOp + arith::OrIOp |
| | 215 | OP_XOR_INT_LIT16 | arith::ConstantIntOp + arith::XOrIOp |
| Instruction22t | 50 | OP_IF_EQ | arith::CmpIOp + cf::CondBranchOp |
| | 51 | OP_IF_NE | arith::CmpIOp + cf::CondBranchOp |
| | 52 | OP_IF_LT | arith::CmpIOp + cf::CondBranchOp |
| | 53 | OP_IF_GE | arith::CmpIOp + cf::CondBranchOp |
| | 54 | OP_IF_GT | arith::CmpIOp + cf::CondBranchOp |
| | 55 | OP_IF_LE | arith::CmpIOp + cf::CondBranchOp |
| Instruction22x | 2 | OP_MOVE_FROM16 | KUNAI::MjolnIR::MoveOp |
| | 5 | OP_MOVE_WIDE_FROM16 | KUNAI::MjolnIR::MoveOp |
| | 8 | OP_MOVE_OBJECT_FROM16 | KUNAI::MjolnIR::MoveOp |
| Instruction23x | 144 | OP_ADD_INT | arith::AddIOp |
| | 155 | OP_ADD_LONG | arith::AddIOp |
| | 166 | OP_ADD_FLOAT | arith::AddFOp |
| | 171 | OP_ADD_DOUBLE | arith::AddFOp |
| | 145 | OP_SUB_INT | arith::SubIOp |
| | 156 | OP_SUB_LONG | arith::SubIOp |
| | 167 | OP_SUB_FLOAT | arith::SubFOp |
| | 172 | OP_SUB_DOUBLE | arith::SubFOp |
| | 146 | OP_MUL_INT | arith::MulIOp |
| | 157 | OP_MUL_LONG | arith::MulIOp |
| | 168 | OP_MUL_FLOAT | arith::MulFOp |
| | 173 | OP_MUL_DOUBLE | arith::MulFOp |
| | 147 | OP_DIV_INT | arith::DivSIOp |
| | 158 | OP_DIV_LONG | arith::DivSIOp |
| | 169 | OP_DIV_FLOAT | arith::DivFOp |
| | 174 | OP_DIV_DOUBLE | arith::DivFOp |
| | 148 | OP_REM_INT | arith::RemSIOp |
| | 159 | OP_REM_LONG | arith::RemSIOp |
| | 170 | OP_REM_FLOAT | arith::RemFOp |
| | 175 | OP_REM_DOUBLE | arith::RemFOp |
| | 149 | OP_AND_INT | arith::AndIOp |
| | 160 | OP_AND_LONG | arith::AndIOp |
| | 150 | OP_OR_INT | arith::OrIOp |
| | 161 | OP_OR_LONG | arith::OrIOp |
| | 151 | OP_XOR_INT | arith::XOrIOp |
| | 162 | OP_XOR_LONG | arith::XOrIOp |
| | 152 | OP_SHL_INT | arith::ShLIOp |
| | 163 | OP_SHL_LONG | arith::ShLIOp |
| | 153 | OP_SHR_INT | arith::ShRSIOp |
| | 164 | OP_SHR_LONG | arith::ShRSIOp |
| | 154 | OP_USHR_INT | arith::ShRUIOp |
| | 165 | OP_USHR_LONG | arith::ShRUIOp |
| Instruction30t | 42 | OP_GOTO_32 | cf::BranchOp |
| Instruction31c | 27 | OP_CONST_STRING_JUMBO | KUNAI::MjolnIR::LoadString |
| Instruction31i | 20 | OP_CONST | arith::ConstantFloatOp |
| | 23 | OP_CONST_WIDE_32 | arith::ConstantFloatOp |
| Instruction32x | 3 | OP_MOVE_16 | KUNAI::MjolnIR::MoveOp |
| | 6 | OP_MOVE_WIDE_16 | KUNAI::MjolnIR::MoveOp |
| | 9 | OP_MOVE_OBJECT_16 | KUNAI::MjolnIR::MoveOp |
| Instruction35c | 110 | OP_INVOKE_VIRTUAL | KUNAI::MjolnIR::InvokeOp |
| | 111 | OP_INVOKE_SUPER | KUNAI::MjolnIR::InvokeOp |
| | 112 | OP_INVOKE_DIRECT | KUNAI::MjolnIR::InvokeOp |
| | 113 | OP_INVOKE_STATIC | KUNAI::MjolnIR::InvokeOp |
| | 114 | OP_INVOKE_INTERFACE | KUNAI::MjolnIR::InvokeOp |
| Instruction51l | 24 | OP_CONST_WIDE | arith::ConstantFloatOp |