// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPrngWH
#define _C_CkPrngWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPrngW CkPrngW_Create(void);
CK_C_VISIBLE_PUBLIC void CkPrngW_Dispose(HCkPrngW handle);
CK_C_VISIBLE_PUBLIC void CkPrngW_getDebugLogFilePath(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPrngW_putDebugLogFilePath(HCkPrngW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_debugLogFilePath(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC void CkPrngW_getLastErrorHtml(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_lastErrorHtml(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC void CkPrngW_getLastErrorText(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_lastErrorText(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC void CkPrngW_getLastErrorXml(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_lastErrorXml(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_getLastMethodSuccess(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPrngW_putLastMethodSuccess(HCkPrngW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPrngW_getPrngName(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPrngW_putPrngName(HCkPrngW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_prngName(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_getVerboseLogging(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPrngW_putVerboseLogging(HCkPrngW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPrngW_getVersion(HCkPrngW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_version(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_AddEntropy(HCkPrngW cHandle, const wchar_t *entropy, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_AddEntropyBytes(HCkPrngW cHandle, HCkByteData entropy);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_ExportEntropy(HCkPrngW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_exportEntropy(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_FirebasePushId(HCkPrngW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_firebasePushId(HCkPrngW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_GenRandom(HCkPrngW cHandle, int numBytes, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_genRandom(HCkPrngW cHandle, int numBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_GenRandomBd(HCkPrngW cHandle, int numBytes, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_GenRandomBytes(HCkPrngW cHandle, int numBytes, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_GetEntropy(HCkPrngW cHandle, int numBytes, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_getEntropy(HCkPrngW cHandle, int numBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_GetEntropyBytes(HCkPrngW cHandle, int numBytes, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_ImportEntropy(HCkPrngW cHandle, const wchar_t *entropy);
CK_C_VISIBLE_PUBLIC int CkPrngW_RandomInt(HCkPrngW cHandle, int low, int high);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_RandomPassword(HCkPrngW cHandle, int length, BOOL mustIncludeDigit, BOOL upperAndLowercase, const wchar_t *mustHaveOneOf, const wchar_t *excludeChars, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_randomPassword(HCkPrngW cHandle, int length, BOOL mustIncludeDigit, BOOL upperAndLowercase, const wchar_t *mustHaveOneOf, const wchar_t *excludeChars);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_RandomString(HCkPrngW cHandle, int length, BOOL bDigits, BOOL bLower, BOOL bUpper, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPrngW_randomString(HCkPrngW cHandle, int length, BOOL bDigits, BOOL bLower, BOOL bUpper);
CK_C_VISIBLE_PUBLIC BOOL CkPrngW_SaveLastError(HCkPrngW cHandle, const wchar_t *path);
#endif
