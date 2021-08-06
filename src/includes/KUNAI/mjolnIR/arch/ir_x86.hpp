/**
 * @file ir_x86.hpp
 * @author Farenain
 * 
 * @brief Values specific for x86 and x86-64 for the registers
 *        this will be enums for the registers, sizes and strings.
 */

#include <map>

namespace KUNAI
{
    namespace MJOLNIR
    {
        enum x86_regs_t
        /**
         * @brief X86 registers, enums for IR, sizes and strings.
         */
        {
            // General purpose registers
            rax, eax, ax, ah, al,
            rbx, ebx, bx, bh, bl,
            rcx, ecx, cx, ch, cl,
            rdx, edx, dx, dh, dl,
            // pointer registers
            rdi, edi, di,
            rsi, esi, si,
            // stack registers
            rbp, ebp, bp,
            rsp, esp, sp,
            // program counter
            rip, eip, ip,
            // extended registers in x86-64
            r8, r8d, r8w, r8b, 
            r9, r9d, r9w, r9b, 
            r10, r10d, r10w, r10b,
            r11, r11d, r11w, r11b, 
            r12, r12d, r12w, r12b, 
            r13, r13d, r13w, r13b, 
            r14, r14d, r14w, r14b, 
            r15, r15d, r15w, r15b, 
            // flags for state representation
            eflags,
            mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7,
            zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, 
            zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, 
            zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, 
            zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, 
            zmm28, zmm29, zmm30, zmm31, mxcsr,
            cr0, cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8, cr9,
            cr10, cr11, cr12, cr13, cr14, cr15,
            cs, ds, es, fs, gs, ss,
            dr0, dr1, dr2, dr3, dr6, dr7
        };

        static const std::map<x86_regs_t, size_t> x86_regs_size = {
            {rax,8}, {eax,4}, {ax,2}, {ah,1}, {al,1},
            {rbx,8}, {ebx,4}, {bx,2}, {bh,1}, {bl,1},
            {rcx,8}, {ecx,4}, {cx,2}, {ch,1}, {cl,1},
            {rdx,8}, {edx,4}, {dx,2}, {dh,1}, {dl,1},
            {rdi,8}, {edi,4}, {di,2},
            {rsi,8}, {esi,4}, {si,2},
            {rip,8}, {eip,4}, {ip,2},
            {r8, 8}, {r8d,4}, {r8w,2}, {r8b,1}, 
            {r9, 8}, {r9d,4}, {r9w,2}, {r9b,1},
            {r10, 8}, {r10d,4}, {r10w,2}, {r10b,1},
            {r11, 8}, {r11d,4}, {r11w,2}, {r11b,1}, 
            {r12, 8}, {r12d,4}, {r12w,2}, {r12b,1},
            {r13, 8}, {r13d,4}, {r13w,2}, {r13b,1},
            {r14, 8}, {r14d,4}, {r14w,2}, {r14b,1},
            {r15, 8}, {r15d,4}, {r15w,2}, {r15b,1}
        };

        static const std::map<x86_regs_t, std::string> x86_regs_name = {
            {rax,"rax"}, {eax,"eax"}, {ax,"ax"}, {ah,"ah"}, {al,"al"},
            {rbx,"rbx"}, {ebx,"ebx"}, {bx,"bx"}, {bh,"bh"}, {bl,"bl"},
            {rcx,"rcx"}, {ecx,"ecx"}, {cx,"cx"}, {ch,"ch"}, {cl,"cl"},
            {rdx,"rdx"}, {edx,"edx"}, {dx,"dx"}, {dh,"dh"}, {dl,"dl"},
            {rdi,"rdi"}, {edi,"edi"}, {di,"di"},
            {rsi,"rsi"}, {esi,"esi"}, {si,"si"},
            {rip,"rip"}, {eip,"eip"}, {ip,"ip"},
            {r8,"r8"}, {r8d,"r8d"}, {r8w,"r8w"}, {r8b,"r8b"}, 
            {r9,"r9"}, {r9d,"r9d"}, {r9w,"r9w"}, {r9b,"r9b"},
            {r10,"r10"}, {r10d,"r10d"}, {r10w,"r10w"}, {r10b,"r10b"},
            {r11,"r11"}, {r11d,"r11d"}, {r11w,"r11w"}, {r11b,"r11b"}, 
            {r12,"r12"}, {r12d,"r12d"}, {r12w,"r12w"}, {r12b,"r12b"},
            {r13,"r13"}, {r13d,"r13d"}, {r13w,"r13w"}, {r13b,"r13b"},
            {r14,"r14"}, {r14d,"r14d"}, {r14w,"r14w"}, {r14b,"r14b"},
            {r15,"r15"}, {r15d,"r15d"}, {r15w,"r15w"}, {r15b,"r15b"}
        };
    }
}