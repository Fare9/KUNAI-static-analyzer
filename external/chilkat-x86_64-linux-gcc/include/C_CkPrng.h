// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPrng_H
#define _C_CkPrng_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPrng CkPrng_Create(void);
CK_C_VISIBLE_PUBLIC void CkPrng_Dispose(HCkPrng handle);
CK_C_VISIBLE_PUBLIC void CkPrng_getDebugLogFilePath(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPrng_putDebugLogFilePath(HCkPrng cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPrng_debugLogFilePath(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_getLastErrorHtml(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPrng_lastErrorHtml(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_getLastErrorText(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPrng_lastErrorText(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_getLastErrorXml(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPrng_lastErrorXml(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_getLastMethodSuccess(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_putLastMethodSuccess(HCkPrng cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPrng_getPrngName(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPrng_putPrngName(HCkPrng cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPrng_prngName(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_getUtf8(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_putUtf8(HCkPrng cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_getVerboseLogging(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC void CkPrng_putVerboseLogging(HCkPrng cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPrng_getVersion(HCkPrng cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPrng_version(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_AddEntropy(HCkPrng cHandle, const char *entropy, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_AddEntropyBytes(HCkPrng cHandle, HCkByteData entropy);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_ExportEntropy(HCkPrng cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_exportEntropy(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_FirebasePushId(HCkPrng cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_firebasePushId(HCkPrng cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_GenRandom(HCkPrng cHandle, int numBytes, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_genRandom(HCkPrng cHandle, int numBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_GenRandomBd(HCkPrng cHandle, int numBytes, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_GenRandomBytes(HCkPrng cHandle, int numBytes, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_GetEntropy(HCkPrng cHandle, int numBytes, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_getEntropy(HCkPrng cHandle, int numBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_GetEntropyBytes(HCkPrng cHandle, int numBytes, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_ImportEntropy(HCkPrng cHandle, const char *entropy);
CK_C_VISIBLE_PUBLIC int CkPrng_RandomInt(HCkPrng cHandle, int low, int high);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_RandomPassword(HCkPrng cHandle, int length, BOOL mustIncludeDigit, BOOL upperAndLowercase, const char *mustHaveOneOf, const char *excludeChars, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_randomPassword(HCkPrng cHandle, int length, BOOL mustIncludeDigit, BOOL upperAndLowercase, const char *mustHaveOneOf, const char *excludeChars);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_RandomString(HCkPrng cHandle, int length, BOOL bDigits, BOOL bLower, BOOL bUpper, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPrng_randomString(HCkPrng cHandle, int length, BOOL bDigits, BOOL bLower, BOOL bUpper);
CK_C_VISIBLE_PUBLIC BOOL CkPrng_SaveLastError(HCkPrng cHandle, const char *path);
#endif
