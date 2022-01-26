// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkStringBuilder_H
#define _C_CkStringBuilder_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkStringBuilder CkStringBuilder_Create(void);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_Dispose(HCkStringBuilder handle);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_getIntValue(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_putIntValue(HCkStringBuilder cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_getIsBase64(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_getLastMethodSuccess(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_putLastMethodSuccess(HCkStringBuilder cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_getLength(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_getUtf8(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_putUtf8(HCkStringBuilder cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Append(HCkStringBuilder cHandle, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendBd(HCkStringBuilder cHandle, HCkBinData binData, const char *charset, int offset, int numBytes);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendEncoded(HCkStringBuilder cHandle, HCkByteData binaryData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendInt(HCkStringBuilder cHandle, int value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendInt64(HCkStringBuilder cHandle, __int64 value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendLine(HCkStringBuilder cHandle, const char *value, BOOL crlf);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_AppendSb(HCkStringBuilder cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_Clear(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Contains(HCkStringBuilder cHandle, const char *str, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ContainsWord(HCkStringBuilder cHandle, const char *word, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ContentsEqual(HCkStringBuilder cHandle, const char *str, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ContentsEqualSb(HCkStringBuilder cHandle, HCkStringBuilder sb, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Decode(HCkStringBuilder cHandle, const char *encoding, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_DecodeAndAppend(HCkStringBuilder cHandle, const char *value, const char *encoding, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Encode(HCkStringBuilder cHandle, const char *encoding, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_EndsWith(HCkStringBuilder cHandle, const char *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_EntityDecode(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetAfterBetween(HCkStringBuilder cHandle, const char *searchAfter, const char *beginMark, const char *endMark, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getAfterBetween(HCkStringBuilder cHandle, const char *searchAfter, const char *beginMark, const char *endMark);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetAfterFinal(HCkStringBuilder cHandle, const char *marker, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getAfterFinal(HCkStringBuilder cHandle, const char *marker, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetAsString(HCkStringBuilder cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getAsString(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetBefore(HCkStringBuilder cHandle, const char *marker, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getBefore(HCkStringBuilder cHandle, const char *marker, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetBetween(HCkStringBuilder cHandle, const char *beginMark, const char *endMark, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getBetween(HCkStringBuilder cHandle, const char *beginMark, const char *endMark);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetDecoded(HCkStringBuilder cHandle, const char *encoding, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetEncoded(HCkStringBuilder cHandle, const char *encoding, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getEncoded(HCkStringBuilder cHandle, const char *encoding, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetNth(HCkStringBuilder cHandle, int index, const char *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getNth(HCkStringBuilder cHandle, int index, const char *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_GetRange(HCkStringBuilder cHandle, int startIndex, int numChars, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_getRange(HCkStringBuilder cHandle, int startIndex, int numChars, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_LastNLines(HCkStringBuilder cHandle, int numLines, BOOL bCrlf, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkStringBuilder_lastNLines(HCkStringBuilder cHandle, int numLines, BOOL bCrlf);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_LoadFile(HCkStringBuilder cHandle, const char *path, const char *charset);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_Obfuscate(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Prepend(HCkStringBuilder cHandle, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_PunyDecode(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_PunyEncode(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_RemoveAfterFinal(HCkStringBuilder cHandle, const char *marker);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_RemoveBefore(HCkStringBuilder cHandle, const char *marker);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_RemoveCharsAt(HCkStringBuilder cHandle, int startIndex, int numChars);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_Replace(HCkStringBuilder cHandle, const char *value, const char *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ReplaceAfterFinal(HCkStringBuilder cHandle, const char *marker, const char *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ReplaceAllBetween(HCkStringBuilder cHandle, const char *beginMark, const char *endMark, const char *replacement, BOOL replaceMarks);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_ReplaceBetween(HCkStringBuilder cHandle, const char *beginMark, const char *endMark, const char *value, const char *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ReplaceFirst(HCkStringBuilder cHandle, const char *value, const char *replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_ReplaceI(HCkStringBuilder cHandle, const char *value, int replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_ReplaceNoCase(HCkStringBuilder cHandle, const char *value, const char *replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilder_ReplaceWord(HCkStringBuilder cHandle, const char *value, const char *replacement);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_SecureClear(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_SetNth(HCkStringBuilder cHandle, int index, const char *value, const char *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_SetString(HCkStringBuilder cHandle, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Shorten(HCkStringBuilder cHandle, int numChars);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_StartsWith(HCkStringBuilder cHandle, const char *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ToCRLF(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ToLF(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ToLowercase(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_ToUppercase(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_Trim(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_TrimInsideSpaces(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC void CkStringBuilder_Unobfuscate(HCkStringBuilder cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_WriteFile(HCkStringBuilder cHandle, const char *path, const char *charset, BOOL emitBom);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilder_WriteFileIfModified(HCkStringBuilder cHandle, const char *path, const char *charset, BOOL emitBom);
#endif
