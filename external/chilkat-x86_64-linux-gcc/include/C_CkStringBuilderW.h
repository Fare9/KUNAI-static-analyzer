// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkStringBuilderWH
#define _C_CkStringBuilderWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkStringBuilderW CkStringBuilderW_Create(void);
CK_C_VISIBLE_PUBLIC void CkStringBuilderW_Dispose(HCkStringBuilderW handle);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_getIntValue(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC void  CkStringBuilderW_putIntValue(HCkStringBuilderW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_getIsBase64(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_getLastMethodSuccess(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC void  CkStringBuilderW_putLastMethodSuccess(HCkStringBuilderW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_getLength(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Append(HCkStringBuilderW cHandle, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendBd(HCkStringBuilderW cHandle, HCkBinDataW binData, const wchar_t *charset, int offset, int numBytes);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendEncoded(HCkStringBuilderW cHandle, HCkByteData binaryData, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendInt(HCkStringBuilderW cHandle, int value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendInt64(HCkStringBuilderW cHandle, __int64 value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendLine(HCkStringBuilderW cHandle, const wchar_t *value, BOOL crlf);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_AppendSb(HCkStringBuilderW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC void CkStringBuilderW_Clear(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Contains(HCkStringBuilderW cHandle, const wchar_t *str, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ContainsWord(HCkStringBuilderW cHandle, const wchar_t *word, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ContentsEqual(HCkStringBuilderW cHandle, const wchar_t *str, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ContentsEqualSb(HCkStringBuilderW cHandle, HCkStringBuilderW sb, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Decode(HCkStringBuilderW cHandle, const wchar_t *encoding, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_DecodeAndAppend(HCkStringBuilderW cHandle, const wchar_t *value, const wchar_t *encoding, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Encode(HCkStringBuilderW cHandle, const wchar_t *encoding, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_EndsWith(HCkStringBuilderW cHandle, const wchar_t *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_EntityDecode(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetAfterBetween(HCkStringBuilderW cHandle, const wchar_t *searchAfter, const wchar_t *beginMark, const wchar_t *endMark, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getAfterBetween(HCkStringBuilderW cHandle, const wchar_t *searchAfter, const wchar_t *beginMark, const wchar_t *endMark);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetAfterFinal(HCkStringBuilderW cHandle, const wchar_t *marker, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getAfterFinal(HCkStringBuilderW cHandle, const wchar_t *marker, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetAsString(HCkStringBuilderW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getAsString(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetBefore(HCkStringBuilderW cHandle, const wchar_t *marker, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getBefore(HCkStringBuilderW cHandle, const wchar_t *marker, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetBetween(HCkStringBuilderW cHandle, const wchar_t *beginMark, const wchar_t *endMark, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getBetween(HCkStringBuilderW cHandle, const wchar_t *beginMark, const wchar_t *endMark);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetDecoded(HCkStringBuilderW cHandle, const wchar_t *encoding, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetEncoded(HCkStringBuilderW cHandle, const wchar_t *encoding, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getEncoded(HCkStringBuilderW cHandle, const wchar_t *encoding, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetNth(HCkStringBuilderW cHandle, int index, const wchar_t *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getNth(HCkStringBuilderW cHandle, int index, const wchar_t *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_GetRange(HCkStringBuilderW cHandle, int startIndex, int numChars, BOOL removeFlag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_getRange(HCkStringBuilderW cHandle, int startIndex, int numChars, BOOL removeFlag);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_LastNLines(HCkStringBuilderW cHandle, int numLines, BOOL bCrlf, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkStringBuilderW_lastNLines(HCkStringBuilderW cHandle, int numLines, BOOL bCrlf);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_LoadFile(HCkStringBuilderW cHandle, const wchar_t *path, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC void CkStringBuilderW_Obfuscate(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Prepend(HCkStringBuilderW cHandle, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_PunyDecode(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_PunyEncode(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_RemoveAfterFinal(HCkStringBuilderW cHandle, const wchar_t *marker);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_RemoveBefore(HCkStringBuilderW cHandle, const wchar_t *marker);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_RemoveCharsAt(HCkStringBuilderW cHandle, int startIndex, int numChars);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_Replace(HCkStringBuilderW cHandle, const wchar_t *value, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ReplaceAfterFinal(HCkStringBuilderW cHandle, const wchar_t *marker, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ReplaceAllBetween(HCkStringBuilderW cHandle, const wchar_t *beginMark, const wchar_t *endMark, const wchar_t *replacement, BOOL replaceMarks);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_ReplaceBetween(HCkStringBuilderW cHandle, const wchar_t *beginMark, const wchar_t *endMark, const wchar_t *value, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ReplaceFirst(HCkStringBuilderW cHandle, const wchar_t *value, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_ReplaceI(HCkStringBuilderW cHandle, const wchar_t *value, int replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_ReplaceNoCase(HCkStringBuilderW cHandle, const wchar_t *value, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC int CkStringBuilderW_ReplaceWord(HCkStringBuilderW cHandle, const wchar_t *value, const wchar_t *replacement);
CK_C_VISIBLE_PUBLIC void CkStringBuilderW_SecureClear(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_SetNth(HCkStringBuilderW cHandle, int index, const wchar_t *value, const wchar_t *delimiterChar, BOOL exceptDoubleQuoted, BOOL exceptEscaped);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_SetString(HCkStringBuilderW cHandle, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Shorten(HCkStringBuilderW cHandle, int numChars);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_StartsWith(HCkStringBuilderW cHandle, const wchar_t *substr, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ToCRLF(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ToLF(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ToLowercase(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_ToUppercase(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_Trim(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_TrimInsideSpaces(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC void CkStringBuilderW_Unobfuscate(HCkStringBuilderW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_WriteFile(HCkStringBuilderW cHandle, const wchar_t *path, const wchar_t *charset, BOOL emitBom);
CK_C_VISIBLE_PUBLIC BOOL CkStringBuilderW_WriteFileIfModified(HCkStringBuilderW cHandle, const wchar_t *path, const wchar_t *charset, BOOL emitBom);
#endif
