// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkSecureStringWH
#define _C_CkSecureStringWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkSecureStringW CkSecureStringW_Create(void);
CK_C_VISIBLE_PUBLIC void CkSecureStringW_Dispose(HCkSecureStringW handle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_getLastMethodSuccess(HCkSecureStringW cHandle);
CK_C_VISIBLE_PUBLIC void  CkSecureStringW_putLastMethodSuccess(HCkSecureStringW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSecureStringW_getMaintainHash(HCkSecureStringW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkSecureStringW_putMaintainHash(HCkSecureStringW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSecureStringW_maintainHash(HCkSecureStringW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_getReadOnly(HCkSecureStringW cHandle);
CK_C_VISIBLE_PUBLIC void  CkSecureStringW_putReadOnly(HCkSecureStringW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_Access(HCkSecureStringW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSecureStringW_access(HCkSecureStringW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_Append(HCkSecureStringW cHandle, const wchar_t *str);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_AppendSb(HCkSecureStringW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_AppendSecure(HCkSecureStringW cHandle, HCkSecureStringW secStr);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_HashVal(HCkSecureStringW cHandle, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkSecureStringW_hashVal(HCkSecureStringW cHandle, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_LoadFile(HCkSecureStringW cHandle, const wchar_t *path, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_SecStrEquals(HCkSecureStringW cHandle, HCkSecureStringW secStr);
CK_C_VISIBLE_PUBLIC BOOL CkSecureStringW_VerifyHash(HCkSecureStringW cHandle, const wchar_t *hashVal, const wchar_t *encoding);
#endif
