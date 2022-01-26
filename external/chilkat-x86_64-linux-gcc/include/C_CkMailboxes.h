// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkMailboxes_H
#define _C_CkMailboxes_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkMailboxes CkMailboxes_Create(void);
CK_C_VISIBLE_PUBLIC void CkMailboxes_Dispose(HCkMailboxes handle);
CK_C_VISIBLE_PUBLIC int CkMailboxes_getCount(HCkMailboxes cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_getLastMethodSuccess(HCkMailboxes cHandle);
CK_C_VISIBLE_PUBLIC void CkMailboxes_putLastMethodSuccess(HCkMailboxes cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_getUtf8(HCkMailboxes cHandle);
CK_C_VISIBLE_PUBLIC void CkMailboxes_putUtf8(HCkMailboxes cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_GetFlags(HCkMailboxes cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkMailboxes_getFlags(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC int CkMailboxes_GetMailboxIndex(HCkMailboxes cHandle, const char *mbxName);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_GetName(HCkMailboxes cHandle, int index, HCkString outStrName);
CK_C_VISIBLE_PUBLIC const char *CkMailboxes_getName(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_GetNthFlag(HCkMailboxes cHandle, int index, int flagIndex, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkMailboxes_getNthFlag(HCkMailboxes cHandle, int index, int flagIndex);
CK_C_VISIBLE_PUBLIC int CkMailboxes_GetNumFlags(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_HasFlag(HCkMailboxes cHandle, int index, const char *flagName);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_HasInferiors(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_IsMarked(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_IsSelectable(HCkMailboxes cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkMailboxes_LoadTaskResult(HCkMailboxes cHandle, HCkTask task);
#endif
