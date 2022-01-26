// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkHashtable_H
#define _C_CkHashtable_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkHashtable CkHashtable_Create(void);
CK_C_VISIBLE_PUBLIC void CkHashtable_Dispose(HCkHashtable handle);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_getLastMethodSuccess(HCkHashtable cHandle);
CK_C_VISIBLE_PUBLIC void CkHashtable_putLastMethodSuccess(HCkHashtable cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_getUtf8(HCkHashtable cHandle);
CK_C_VISIBLE_PUBLIC void CkHashtable_putUtf8(HCkHashtable cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_AddFromXmlSb(HCkHashtable cHandle, HCkStringBuilder sbXml);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_AddInt(HCkHashtable cHandle, const char *key, int value);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_AddQueryParams(HCkHashtable cHandle, const char *queryParams);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_AddStr(HCkHashtable cHandle, const char *key, const char *value);
CK_C_VISIBLE_PUBLIC void CkHashtable_Clear(HCkHashtable cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_ClearWithNewCapacity(HCkHashtable cHandle, int capacity);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_Contains(HCkHashtable cHandle, const char *key);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_ContainsIntKey(HCkHashtable cHandle, int key);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_GetKeys(HCkHashtable cHandle, HCkStringTable strTable);
CK_C_VISIBLE_PUBLIC int CkHashtable_LookupInt(HCkHashtable cHandle, const char *key);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_LookupStr(HCkHashtable cHandle, const char *key, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkHashtable_lookupStr(HCkHashtable cHandle, const char *key);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_Remove(HCkHashtable cHandle, const char *key);
CK_C_VISIBLE_PUBLIC BOOL CkHashtable_ToXmlSb(HCkHashtable cHandle, HCkStringBuilder sbXml);
#endif
