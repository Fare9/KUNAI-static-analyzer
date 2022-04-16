// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAzureStorage_H
#define _C_CkAuthAzureStorage_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAuthAzureStorage CkAuthAzureStorage_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_Dispose(HCkAuthAzureStorage handle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getAccessKey(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putAccessKey(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_accessKey(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getAccount(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putAccount(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_account(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getDebugLogFilePath(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putDebugLogFilePath(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_debugLogFilePath(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getLastErrorHtml(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_lastErrorHtml(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getLastErrorText(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_lastErrorText(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getLastErrorXml(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_lastErrorXml(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureStorage_getLastMethodSuccess(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putLastMethodSuccess(HCkAuthAzureStorage cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getScheme(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putScheme(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_scheme(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getService(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putService(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_service(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureStorage_getUtf8(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putUtf8(HCkAuthAzureStorage cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureStorage_getVerboseLogging(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putVerboseLogging(HCkAuthAzureStorage cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getVersion(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_version(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_getXMsVersion(HCkAuthAzureStorage cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureStorage_putXMsVersion(HCkAuthAzureStorage cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureStorage_xMsVersion(HCkAuthAzureStorage cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureStorage_SaveLastError(HCkAuthAzureStorage cHandle, const char *path);
#endif
