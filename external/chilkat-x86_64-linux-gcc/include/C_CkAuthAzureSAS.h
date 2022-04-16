// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAzureSAS_H
#define _C_CkAuthAzureSAS_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAuthAzureSAS CkAuthAzureSAS_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_Dispose(HCkAuthAzureSAS handle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getAccessKey(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putAccessKey(HCkAuthAzureSAS cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_accessKey(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getDebugLogFilePath(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putDebugLogFilePath(HCkAuthAzureSAS cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_debugLogFilePath(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getLastErrorHtml(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_lastErrorHtml(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getLastErrorText(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_lastErrorText(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getLastErrorXml(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_lastErrorXml(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_getLastMethodSuccess(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putLastMethodSuccess(HCkAuthAzureSAS cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getStringToSign(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putStringToSign(HCkAuthAzureSAS cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_stringToSign(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_getUtf8(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putUtf8(HCkAuthAzureSAS cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_getVerboseLogging(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_putVerboseLogging(HCkAuthAzureSAS cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_getVersion(HCkAuthAzureSAS cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_version(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureSAS_Clear(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_GenerateToken(HCkAuthAzureSAS cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureSAS_generateToken(HCkAuthAzureSAS cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_SaveLastError(HCkAuthAzureSAS cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_SetNonTokenParam(HCkAuthAzureSAS cHandle, const char *name, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureSAS_SetTokenParam(HCkAuthAzureSAS cHandle, const char *name, const char *authParamName, const char *value);
#endif
