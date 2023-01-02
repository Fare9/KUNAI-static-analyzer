| Opcode | Opcode name | MjolnIR |
|:------:|:------------|:--------|
| 0x00 | nop | IRNop |
| 0x01 | move vx,vy | IRAssign |
| 0x02 | move/from16 | IRAssign |
| 0x03 | move/16 | IRAssign |
| 0x04 | move-wide | IRAssign |
| 0x05 | move-wide/from16 vx,vy | IRAssign |
| 0x06 | move-wide/16 | IRAssign |
| 0x07 | move-object vx,vy | IRAssign |
| 0x08 | move-object/from16 vx,vy | IRAssign |
| 0x09 | move-object/16 | IRAssign |
| 0x0A | move-result vx | Assigned as result to IRCall |
| 0x0B | move-result-wide vx | Assigned as result to IRCall |
| 0x0C | move-result-object vx | Assigned as result to IRCall |
| 0x0D | move-exception vx |   |
| 0x0E | return-void | IRRet |
| 0x0F | return vx | IRRet |
| 0x10 | return-wide vx | IRRet |
| 0x11 | return-object vx | IRRet |
| 0x12 | const/4 vx,lit4 | IRAssign |
| 0x13 | const/16 vx,lit16 | IRAssign |
| 0x14 | const vx, lit32 | IRAssign |
| 0x15 | const/high16 v0, lit16 | IRAssign |
| 0x16 | const-wide/16 vx, lit16 | IRAssign |
| 0x17 | const-wide/32 vx, lit32 | IRAssign |
| 0x18 | const-wide vx, lit64 | IRAssign |
| 0x19 | const-wide/high16 vx,lit16 | IRAssign |
| 0x1A | const-string vx,string id | IRAssign |
| 0x1B | const-string-jumbo vx,string | IRAssign |
| 0x1C | const-class vx,type id | IRAssign |
| 0x1D | monitor-enter vx |   |
| 0x1E | monitor-exit vx |   |
| 0x1F | check-cast vx, type id |   |
| 0x20 | instance-of vx,vy,type id |   |
| 0x21 | array-length vx,vy |   |
| 0x22 | new-instance vx,type | IRNew |
| 0x23 | new-array vx,vy,type id | IRAlloca |
| 0x24 | filled-new-array {parameters},type id |   |
| 0x25 | filled-new-array-range {vx..vy},type id |   |
| 0x26 | fill-array-data vx,array\_data\_offset |   |
| 0x27 | throw vx |   |
| 0x28 | goto target | IRUJmp |
| 0x29 | goto/16 target | IRUJmp |
| 0x2A | goto/32 target | IRUJmp |
| 0x2B | packed-switch vx,table | IRSwitch |
| 0x2C | sparse-switch vx,table | IRSwitch |
| 0x2D | cmpl-float | IRUnaryOp (CAST\_OP) + IRUnaryOp (CAST\_OP) + IRBComp  |
| 0x2E | cmpg-float vx, vy, vz | IRUnaryOp (CAST\_OP) + IRUnaryOp (CAST\_OP) + IRBComp  |
| 0x2F | cmpl-double vx,vy,vz | IRUnaryOp (CAST\_OP) + IRUnaryOp (CAST\_OP) + IRBComp  |
| 0x30 | cmpg-double vx, vy, vz | IRUnaryOp (CAST\_OP) + IRUnaryOp (CAST\_OP) + IRBComp  |
| 0x31 | cmp-long vx, vy, vz | IRUnaryOp (CAST\_OP) + IRUnaryOp (CAST\_OP) + IRBComp  |
| 0x32 | if-eq vx,vy,target | IRBComp + IRCJmp |
| 0x33 | if-ne vx,vy,target | IRBComp + IRCJmp |
| 0x34 | if-lt vx,vy,target | IRBComp + IRCJmp |
| 0x35 | if-ge vx, vy,target | IRBComp + IRCJmp |
| 0x36 | if-gt vx,vy,target | IRBComp + IRCJmp |
| 0x37 | if-le vx,vy,target | IRBComp + IRCJmp |
| 0x38 | if-eqz vx,target | IRZComp + IRCJmp |
| 0x39 | if-nez vx,target | IRZComp + IRCJmp |
| 0x3A | if-ltz vx,target | IRZComp + IRCJmp |
| 0x3B | if-gez vx,target | IRZComp + IRCJmp |
| 0x3C | if-gtz vx,target | IRZComp + IRCJmp |
| 0x3D | if-lez vx,target | IRZComp + IRCJmp |
| 0x3E | unused 3E |   |
| 0x3F | unused 3F |   |
| 0x40 | unused 40 |   |
| 0x41 | unused 41 |   |
| 0x42 | unused 42 |   |
| 0x43 | unused 43 |   |
| 0x44 | aget vx,vy,vz | IRLoad  |
| 0x45 | aget-wide vx,vy,vz | IRLoad  |
| 0x46 | aget-object vx,vy,vz | IRLoad + IRUnaryOp (CAST\_OP) |
| 0x47 | aget-boolean vx,vy,vz | IRLoad + IRUnaryOp (CAST\_OP) |
| 0x48 | aget-byte vx,vy,vz | IRLoad + IRUnaryOp (CAST\_OP)  |
| 0x49 | aget-char vx, vy,vz | IRLoad + IRUnaryOp (CAST\_OP) |
| 0x4A | aget-short vx,vy,vz | IRLoad + IRUnaryOp (CAST\_OP) |
| 0x4B | aput vx,vy,vz | IRStore |
| 0x4C | aput-wide vx,vy,vz | IRStore |
| 0x4D | aput-object vx,vy,vz | IRStore |
| 0x4E | aput-boolean vx,vy,vz | IRStore |
| 0x4F | aput-byte vx,vy,vz | IRStore |
| 0x50 | aput-char vx,vy,vz | IRStore |
| 0x51 | aput-short vx,vy,vz | IRStore |
| 0x52 | iget vx, vy, field\_id | IRAssign |
| 0x53 | iget-wide vx,vy,field\_id | IRAssign |
| 0x54 | iget-object vx,vy,field\_id | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x55 | iget-boolean vx,vy,field\_id | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x56 | iget-byte vx,vy,field\_id | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x57 | iget-char vx,vy,field\_id | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x58 | iget-short vx,vy,field\_id | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x59 | iput vx,vy, field\_id | IRAssign |
| 0x5A | iput-wide vx,vy, field\_id | IRAssign |
| 0x5B | iput-object vx,vy,field\_id | IRAssign |
| 0x5C | iput-boolean vx,vy, field\_id | IRAssign |
| 0x5D | iput-byte vx,vy,field\_id  | IRAssign |
| 0x5E | iput-char vx,vy,field\_id  | IRAssign |
| 0x5F | iput-short vx,vy,field\_id  | IRAssign |
| 0x60 | sget vx,field\_id  | IRAssign |
| 0x61 | sget-wide vx, field\_id  | IRAssign |
| 0x62 | sget-object vx,field\_id  | IRAssign |
| 0x63 | sget-boolean vx,field\_id  | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x64 | sget-byte vx,field\_id  | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x65 | sget-char vx,field\_id  | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x66 | sget-short vx,field\_id  | IRAssign + IRUnaryOp (CAST\_OP) |
| 0x67 | sput vx, field\_id  | IRAssign |
| 0x68 | sput-wide vx, field\_id  | IRAssign |
| 0x69 | sput-object vx,field\_id  | IRAssign |
| 0x6A | sput-boolean vx,field\_id  | IRAssign |
| 0x6B | sput-byte vx,field\_id  | IRAssign |
| 0x6C | sput-char vx,field\_id  | IRAssign |
| 0x6D | sput-short vx,field\_id  | IRAssign |
| 0x6E | invoke-virtual {parameters },methodtocall | IRCall + IRCallee |
| 0x6F | invoke-super {parameter},methodtocall | IRCall + IRCallee |
| 0x70 | invoke-direct {parameters },methodtocall | IRCall + IRCallee |
| 0x71 | invoke-static {parameters},methodtocall | IRCall + IRCallee |
| 0x72 | invoke-interface {parameters},methodtocall | IRCall + IRCallee |
| 0x73 | unused 73 |   |
| 0x74 | invoke-virtual/range {vx..vy},methodtocall | IRCall + IRCallee |
| 0x75 | invoke-super/range invoke-special | IRCall + IRCallee |
| 0x76 | invoke-direct/range {vx..vy},methodtocall | IRCall + IRCallee |
| 0x77 | invoke-static/range {vx..vy},methodtocall | IRCall + IRCallee |
| 0x78 | invoke-interface-range invoke-interface | IRCall + IRCallee |
| 0x79 | unused 79 |   |
| 0x7A | unused 7A |   |
| 0x7B | neg-int vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x7C | not-int vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x7D | neg-long vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x7E | not-long vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x7F | neg-float vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x80 | neg-double vx,vy  | IRUnaryOp + IRUnaryOp (CAST\_OP)  |
| 0x81 | int-to-long vx, vy  | IRUnaryOp  |
| 0x82 | int-to-float vx, vy  | IRUnaryOp  |
| 0x83 | int-to-double vx, vy  | IRUnaryOp  |
| 0x84 | long-to-int vx,vy  | IRUnaryOp  |
| 0x85 | long-to-float vx, vy  | IRUnaryOp  |
| 0x86 | long-to-double vx, vy | IRUnaryOp |
| 0x87 | float-to-int vx, vy | IRUnaryOp |
| 0x88 | float-to-long vx,vy | IRUnaryOp |
| 0x89 | float-to-double vx, vy | IRUnaryOp |
| 0x8A | double-to-int vx, vy | IRUnaryOp |
| 0x8B | double-to-long vx, vy | IRUnaryOp |
| 0x8C | double-to-float vx, vy | IRUnaryOp |
| 0x8D | int-to-byte vx,vy | IRUnaryOp |
| 0x8E | int-to-char vx,vy | IRUnaryOp |
| 0x8F | int-to-short vx,vy | IRUnaryOp |
| 0x90 | add-int vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x91 | sub-int vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x92 | mul-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x93 | div-int vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x94 | rem-int vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x95 | and-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x96 | or-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x97 | xor-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x98 | shl-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x99 | shr-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9A | ushr-int vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9B | add-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9C | sub-long vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9D | mul-long vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9E | div-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0x9F | rem-long vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA0 | and-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA1 | or-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA2 | xor-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA3 | shl-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA4 | shr-long vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA5 | ushr-long vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA6 | add-float vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA7 | sub-float vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA8 | mul-float vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xA9 | div-float vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAA | rem-float vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAB | add-double vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAC | sub-double vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAD | mul-double vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAE | div-double vx, vy, vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xAF | rem-double vx,vy,vz | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB0 | add-int/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB1 | sub-int/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB2 | mul-int/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB3 | div-int/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB4 | rem-int/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB5 | and-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB6 | or-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB7 | xor-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB8 | shl-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xB9 | shr-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBA | ushr-int/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBB | add-long/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBC | sub-long/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBD | mul-long/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBE | div-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xBF | rem-long/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC0 | and-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC1 | or-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC2 | xor-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC3 | shl-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC4 | shr-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC5 | ushr-long/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC6 | add-float/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC7 | sub-float/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC8 | mul-float/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xC9 | div-float/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCA | rem-float/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCB | add-double/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCC | sub-double/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCD | mul-double/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCE | div-double/2addr vx, vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xCF | rem-double/2addr vx,vy | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD0 | add-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD1 | sub-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD2 | mul-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD3 | div-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD4 | rem-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD5 | and-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD6 | or-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD7 | xor-int/lit16 vx,vy,lit16 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD8 | add-int/lit8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xD9 | sub-int/lit8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDA | mul-int/lit-8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDB | div-int/lit8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDC | rem-int/lit8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDD | and-int/lit8 vx,vy,lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDE | or-int/lit8 vx, vy, lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xDF | xor-int/lit8 vx, vy, lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xE0 | shl-int/lit8 vx, vy, lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xE1 | shr-int/lit8 vx, vy, lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xE2 | ushr-int/lit8 vx, vy, lit8 | IRBinOp + IRUnaryOp (CAST\_OP)  |
| 0xE3 | unused E3 |   |
| 0xE4 | unused E4 |   |
| 0xE5 | unused E5 |   |
| 0xE6 | unused E6 |   |
| 0xE7 | unused E7 |   |
| 0xE8 | unused E8  |   |
| 0xE9 | unused E9  |   |
| 0xEA | unused EA  |   |
| 0xEB | unused EB  |   |
| 0xEC | unused EC  |   |
| 0xED | unused ED  |   |
| 0xEE | execute-inline {parameters},inline ID |   |
| 0xEF | unused EF  |   |
| 0xF0 | invoke-direct-empty |   |
| 0xF1 | unused F1  |   |
| 0xF2 | iget-quick vx,vy,offset  |   |
| 0xF3 | iget-wide-quick vx,vy,offset  |   |
| 0xF4 | iget-object-quick vx,vy,offset  |   |
| 0xF5 | iput-quick vx,vy,offset  |   |
| 0xF6 | iput-wide-quick vx,vy,offset  |   |
| 0xF7 | iput-object-quick vx,vy,offset  |   |
| 0xF8 | invoke-virtual-quick {parameters},vtable offset |   |
| 0xF9 | invoke-virtual-quick/range {parameter range},vtable offset |   |
| 0xFA | invoke-super-quick {parameters},vtable offset |   |
| 0xFB | invoke-super-quick/range {register range},vtable offset |   |
| 0xFC | unused FC  |   |
| 0xFD | unused FD  |   |
| 0xFE | unused FE  |   |
| 0xFF | unused FF  |   |