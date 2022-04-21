// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkStringTableWH
#define _C_CkStringTableWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkStringTableW CkStringTableW_Create(void);
CK_C_VISIBLE_PUBLIC void CkStringTableW_Dispose(HCkStringTableW handle);
CK_C_VISIBLE_PUBLIC int CkStringTableW_getCount(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTableW_getDebugLogFilePath(HCkStringTableW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkStringTableW_putDebugLogFilePath(HCkStringTableW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_debugLogFilePath(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTableW_getLastErrorHtml(HCkStringTableW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_lastErrorHtml(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTableW_getLastErrorText(HCkStringTableW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_lastErrorText(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTableW_getLastErrorXml(HCkStringTableW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_lastErrorXml(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_getLastMethodSuccess(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void  CkStringTableW_putLastMethodSuccess(HCkStringTableW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_getVerboseLogging(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC void  CkStringTableW_putVerboseLogging(HCkStringTableW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkStringTableW_getVersion(HCkStringTableW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_version(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_Append(HCkStringTableW cHandle, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_AppendFromFile(HCkStringTableW cHandle, int maxLineLen, const wchar_t *charset, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_AppendFromSb(HCkStringTableW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC void CkStringTableW_Clear(HCkStringTableW cHandle);
CK_C_VISIBLE_PUBLIC int CkStringTableW_FindSubstring(HCkStringTableW cHandle, int startIndex, const wchar_t *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_GetStrings(HCkStringTableW cHandle, int startIdx, int count, BOOL crlf, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_getStrings(HCkStringTableW cHandle, int startIdx, int count, BOOL crlf);
CK_C_VISIBLE_PUBLIC int CkStringTableW_IntAt(HCkStringTableW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_SaveLastError(HCkStringTableW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_SaveToFile(HCkStringTableW cHandle, const wchar_t *charset, BOOL bCrlf, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_Sort(HCkStringTableW cHandle, BOOL ascending, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_SplitAndAppend(HCkStringTableW cHandle, const wchar_t *inStr, const wchar_t *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringTableW_StringAt(HCkStringTableW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringTableW_stringAt(HCkStringTableW cHandle, int index);
#endif
