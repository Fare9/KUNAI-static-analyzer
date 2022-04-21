// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAzureADWH
#define _C_CkAuthAzureADWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_setAbortCheck(HCkAuthAzureADW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_setPercentDone(HCkAuthAzureADW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_setProgressInfo(HCkAuthAzureADW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_setTaskCompleted(HCkAuthAzureADW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkAuthAzureADW CkAuthAzureADW_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_Dispose(HCkAuthAzureADW handle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getAccessToken(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putAccessToken(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_accessToken(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getClientId(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putClientId(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_clientId(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getClientSecret(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putClientSecret(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_clientSecret(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getDebugLogFilePath(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putDebugLogFilePath(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_debugLogFilePath(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getLastErrorHtml(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_lastErrorHtml(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getLastErrorText(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_lastErrorText(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getLastErrorXml(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_lastErrorXml(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_getLastMethodSuccess(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putLastMethodSuccess(HCkAuthAzureADW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAuthAzureADW_getNumSecondsRemaining(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getResource(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putResource(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_resource(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getTenantId(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putTenantId(HCkAuthAzureADW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_tenantId(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_getValid(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_getVerboseLogging(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthAzureADW_putVerboseLogging(HCkAuthAzureADW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureADW_getVersion(HCkAuthAzureADW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAzureADW_version(HCkAuthAzureADW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_LoadTaskCaller(HCkAuthAzureADW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_ObtainAccessToken(HCkAuthAzureADW cHandle, HCkSocketW connection);
CK_C_VISIBLE_PUBLIC HCkTaskW CkAuthAzureADW_ObtainAccessTokenAsync(HCkAuthAzureADW cHandle, HCkSocketW connection);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureADW_SaveLastError(HCkAuthAzureADW cHandle, const wchar_t *path);
#endif
