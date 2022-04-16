// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkSecureString_H
#define _C_CkSecureString_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkSecureString CkSecureString_Create(void);
CK_C_VISIBLE_PUBLIC void CkSecureString_Dispose(HCkSecureString handle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_getLastMethodSuccess(HCkSecureString cHandle);
CK_C_VISIBLE_PUBLIC void CkSecureString_putLastMethodSuccess(HCkSecureString cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkSecureString_getMaintainHash(HCkSecureString cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkSecureString_putMaintainHash(HCkSecureString cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkSecureString_maintainHash(HCkSecureString cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_getReadOnly(HCkSecureString cHandle);
CK_C_VISIBLE_PUBLIC void CkSecureString_putReadOnly(HCkSecureString cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_getUtf8(HCkSecureString cHandle);
CK_C_VISIBLE_PUBLIC void CkSecureString_putUtf8(HCkSecureString cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_Access(HCkSecureString cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkSecureString_access(HCkSecureString cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_Append(HCkSecureString cHandle, const char *str);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_AppendSb(HCkSecureString cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_AppendSecure(HCkSecureString cHandle, HCkSecureString secStr);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_HashVal(HCkSecureString cHandle, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkSecureString_hashVal(HCkSecureString cHandle, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_LoadFile(HCkSecureString cHandle, const char *path, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_SecStrEquals(HCkSecureString cHandle, HCkSecureString secStr);
CK_C_VISIBLE_PUBLIC BOOL CkSecureString_VerifyHash(HCkSecureString cHandle, const char *hashVal, const char *encoding);
#endif
