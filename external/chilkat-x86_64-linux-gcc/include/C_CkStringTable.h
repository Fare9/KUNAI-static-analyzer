// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkStringTable_H
#define _C_CkStringTable_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkStringTable CkStringTable_Create(void);
CK_C_VISIBLE_PUBLIC void CkStringTable_Dispose(HCkStringTable handle);
CK_C_VISIBLE_PUBLIC int CkStringTable_getCount(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_getDebugLogFilePath(HCkStringTable cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkStringTable_putDebugLogFilePath(HCkStringTable cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_debugLogFilePath(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_getLastErrorHtml(HCkStringTable cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_lastErrorHtml(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_getLastErrorText(HCkStringTable cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_lastErrorText(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_getLastErrorXml(HCkStringTable cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_lastErrorXml(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_getLastMethodSuccess(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_putLastMethodSuccess(HCkStringTable cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_getUtf8(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_putUtf8(HCkStringTable cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_getVerboseLogging(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC void CkStringTable_putVerboseLogging(HCkStringTable cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkStringTable_getVersion(HCkStringTable cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_version(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_Append(HCkStringTable cHandle, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_AppendFromFile(HCkStringTable cHandle, int maxLineLen, const char *charset, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_AppendFromSb(HCkStringTable cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC void CkStringTable_Clear(HCkStringTable cHandle);
CK_C_VISIBLE_PUBLIC int CkStringTable_FindSubstring(HCkStringTable cHandle, int startIndex, const char *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_GetStrings(HCkStringTable cHandle, int startIdx, int count, BOOL crlf, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_getStrings(HCkStringTable cHandle, int startIdx, int count, BOOL crlf);
CK_C_VISIBLE_PUBLIC int CkStringTable_IntAt(HCkStringTable cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_SaveLastError(HCkStringTable cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_SaveToFile(HCkStringTable cHandle, const char *charset, BOOL bCrlf, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_Sort(HCkStringTable cHandle, BOOL ascending, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_SplitAndAppend(HCkStringTable cHandle, const char *inStr, const char *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringTable_StringAt(HCkStringTable cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringTable_stringAt(HCkStringTable cHandle, int index);
#endif
