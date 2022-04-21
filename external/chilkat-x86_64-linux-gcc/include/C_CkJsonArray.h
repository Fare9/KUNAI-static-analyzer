// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJsonArray_H
#define _C_CkJsonArray_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJsonArray CkJsonArray_Create(void);
CK_C_VISIBLE_PUBLIC void CkJsonArray_Dispose(HCkJsonArray handle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_getDebugLogFilePath(HCkJsonArray cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putDebugLogFilePath(HCkJsonArray cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_debugLogFilePath(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_getEmitCompact(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putEmitCompact(HCkJsonArray cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_getEmitCrlf(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putEmitCrlf(HCkJsonArray cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJsonArray_getLastErrorHtml(HCkJsonArray cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_lastErrorHtml(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_getLastErrorText(HCkJsonArray cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_lastErrorText(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_getLastErrorXml(HCkJsonArray cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_lastErrorXml(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_getLastMethodSuccess(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putLastMethodSuccess(HCkJsonArray cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJsonArray_getSize(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_getUtf8(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putUtf8(HCkJsonArray cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_getVerboseLogging(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArray_putVerboseLogging(HCkJsonArray cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJsonArray_getVersion(HCkJsonArray cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_version(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddArrayAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddBoolAt(HCkJsonArray cHandle, int index, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddIntAt(HCkJsonArray cHandle, int index, int value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddNullAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddNumberAt(HCkJsonArray cHandle, int index, const char *numericStr);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddObjectAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddObjectCopyAt(HCkJsonArray cHandle, int index, HCkJsonObject jsonObj);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AddStringAt(HCkJsonArray cHandle, int index, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_AppendArrayItems(HCkJsonArray cHandle, HCkJsonArray jarr);
CK_C_VISIBLE_PUBLIC HCkJsonArray CkJsonArray_ArrayAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_BoolAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC void CkJsonArray_Clear(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_DateAt(HCkJsonArray cHandle, int index, HCkDateTime dateTime);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_DeleteAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_DtAt(HCkJsonArray cHandle, int index, BOOL bLocal, HCkDtObj dt);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_Emit(HCkJsonArray cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_emit(HCkJsonArray cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_EmitSb(HCkJsonArray cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC int CkJsonArray_FindObject(HCkJsonArray cHandle, const char *name, const char *value, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC int CkJsonArray_FindString(HCkJsonArray cHandle, const char *value, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC int CkJsonArray_IntAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_IsNullAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_Load(HCkJsonArray cHandle, const char *jsonArray);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_LoadSb(HCkJsonArray cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC HCkJsonObject CkJsonArray_ObjectAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SaveLastError(HCkJsonArray cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SetBoolAt(HCkJsonArray cHandle, int index, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SetIntAt(HCkJsonArray cHandle, int index, int value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SetNullAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SetNumberAt(HCkJsonArray cHandle, int index, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_SetStringAt(HCkJsonArray cHandle, int index, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_StringAt(HCkJsonArray cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJsonArray_stringAt(HCkJsonArray cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArray_Swap(HCkJsonArray cHandle, int index1, int index2);
CK_C_VISIBLE_PUBLIC int CkJsonArray_TypeAt(HCkJsonArray cHandle, int index);
#endif
