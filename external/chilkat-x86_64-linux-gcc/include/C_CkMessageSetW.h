// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkMessageSetWH
#define _C_CkMessageSetWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkMessageSetW CkMessageSetW_Create(void);
CK_C_VISIBLE_PUBLIC void CkMessageSetW_Dispose(HCkMessageSetW handle);
CK_C_VISIBLE_PUBLIC int CkMessageSetW_getCount(HCkMessageSetW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_getHasUids(HCkMessageSetW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMessageSetW_putHasUids(HCkMessageSetW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_getLastMethodSuccess(HCkMessageSetW cHandle);
CK_C_VISIBLE_PUBLIC void  CkMessageSetW_putLastMethodSuccess(HCkMessageSetW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_ContainsId(HCkMessageSetW cHandle, int msgId);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_FromCompactString(HCkMessageSetW cHandle, const wchar_t *str);
CK_C_VISIBLE_PUBLIC int CkMessageSetW_GetId(HCkMessageSetW cHandle, int index);
CK_C_VISIBLE_PUBLIC void CkMessageSetW_InsertId(HCkMessageSetW cHandle, int id);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_LoadTaskResult(HCkMessageSetW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC void CkMessageSetW_RemoveId(HCkMessageSetW cHandle, int id);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_ToCommaSeparatedStr(HCkMessageSetW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMessageSetW_toCommaSeparatedStr(HCkMessageSetW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSetW_ToCompactString(HCkMessageSetW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkMessageSetW_toCompactString(HCkMessageSetW cHandle);
#endif
