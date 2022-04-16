// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJsonArrayWH
#define _C_CkJsonArrayWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJsonArrayW CkJsonArrayW_Create(void);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_Dispose(HCkJsonArrayW handle);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_getDebugLogFilePath(HCkJsonArrayW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkJsonArrayW_putDebugLogFilePath(HCkJsonArrayW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_debugLogFilePath(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_getEmitCompact(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJsonArrayW_putEmitCompact(HCkJsonArrayW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_getEmitCrlf(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJsonArrayW_putEmitCrlf(HCkJsonArrayW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_getLastErrorHtml(HCkJsonArrayW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_lastErrorHtml(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_getLastErrorText(HCkJsonArrayW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_lastErrorText(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_getLastErrorXml(HCkJsonArrayW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_lastErrorXml(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_getLastMethodSuccess(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJsonArrayW_putLastMethodSuccess(HCkJsonArrayW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJsonArrayW_getSize(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_getVerboseLogging(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJsonArrayW_putVerboseLogging(HCkJsonArrayW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_getVersion(HCkJsonArrayW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_version(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddArrayAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddBoolAt(HCkJsonArrayW cHandle, int index, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddIntAt(HCkJsonArrayW cHandle, int index, int value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddNullAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddNumberAt(HCkJsonArrayW cHandle, int index, const wchar_t *numericStr);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddObjectAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddObjectCopyAt(HCkJsonArrayW cHandle, int index, HCkJsonObjectW jsonObj);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AddStringAt(HCkJsonArrayW cHandle, int index, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_AppendArrayItems(HCkJsonArrayW cHandle, HCkJsonArrayW jarr);
CK_C_VISIBLE_PUBLIC HCkJsonArrayW CkJsonArrayW_ArrayAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_BoolAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC void CkJsonArrayW_Clear(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_DateAt(HCkJsonArrayW cHandle, int index, HCkDateTimeW dateTime);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_DeleteAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_DtAt(HCkJsonArrayW cHandle, int index, BOOL bLocal, HCkDtObjW dt);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_Emit(HCkJsonArrayW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_emit(HCkJsonArrayW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_EmitSb(HCkJsonArrayW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC int CkJsonArrayW_FindObject(HCkJsonArrayW cHandle, const wchar_t *name, const wchar_t *value, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC int CkJsonArrayW_FindString(HCkJsonArrayW cHandle, const wchar_t *value, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC int CkJsonArrayW_IntAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_IsNullAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_Load(HCkJsonArrayW cHandle, const wchar_t *jsonArray);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_LoadSb(HCkJsonArrayW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkJsonArrayW_ObjectAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SaveLastError(HCkJsonArrayW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SetBoolAt(HCkJsonArrayW cHandle, int index, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SetIntAt(HCkJsonArrayW cHandle, int index, int value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SetNullAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SetNumberAt(HCkJsonArrayW cHandle, int index, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_SetStringAt(HCkJsonArrayW cHandle, int index, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_StringAt(HCkJsonArrayW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJsonArrayW_stringAt(HCkJsonArrayW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJsonArrayW_Swap(HCkJsonArrayW cHandle, int index1, int index2);
CK_C_VISIBLE_PUBLIC int CkJsonArrayW_TypeAt(HCkJsonArrayW cHandle, int index);
#endif
