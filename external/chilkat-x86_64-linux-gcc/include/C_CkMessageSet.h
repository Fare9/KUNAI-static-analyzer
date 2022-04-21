// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkMessageSet_H
#define _C_CkMessageSet_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkMessageSet CkMessageSet_Create(void);
CK_C_VISIBLE_PUBLIC void CkMessageSet_Dispose(HCkMessageSet handle);
CK_C_VISIBLE_PUBLIC int CkMessageSet_getCount(HCkMessageSet cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_getHasUids(HCkMessageSet cHandle);
CK_C_VISIBLE_PUBLIC void CkMessageSet_putHasUids(HCkMessageSet cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_getLastMethodSuccess(HCkMessageSet cHandle);
CK_C_VISIBLE_PUBLIC void CkMessageSet_putLastMethodSuccess(HCkMessageSet cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_getUtf8(HCkMessageSet cHandle);
CK_C_VISIBLE_PUBLIC void CkMessageSet_putUtf8(HCkMessageSet cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_ContainsId(HCkMessageSet cHandle, int msgId);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_FromCompactString(HCkMessageSet cHandle, const char *str);
CK_C_VISIBLE_PUBLIC int CkMessageSet_GetId(HCkMessageSet cHandle, int index);
CK_C_VISIBLE_PUBLIC void CkMessageSet_InsertId(HCkMessageSet cHandle, int id);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_LoadTaskResult(HCkMessageSet cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC void CkMessageSet_RemoveId(HCkMessageSet cHandle, int id);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_ToCommaSeparatedStr(HCkMessageSet cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkMessageSet_toCommaSeparatedStr(HCkMessageSet cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMessageSet_ToCompactString(HCkMessageSet cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkMessageSet_toCompactString(HCkMessageSet cHandle);
#endif
