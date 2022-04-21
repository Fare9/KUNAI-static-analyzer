// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAzureAD_H
#define _C_CkAuthAzureAD_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setAbortCheck(HCkAuthAzureAD cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setPercentDone(HCkAuthAzureAD cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setProgressInfo(HCkAuthAzureAD cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setTaskCompleted(HCkAuthAzureAD cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setAbortCheck2(HCkAuthAzureAD cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setPercentDone2(HCkAuthAzureAD cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setProgressInfo2(HCkAuthAzureAD cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setTaskCompleted2(HCkAuthAzureAD cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setExternalProgress(HCkAuthAzureAD cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_setCallbackContext(HCkAuthAzureAD cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkAuthAzureAD CkAuthAzureAD_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_Dispose(HCkAuthAzureAD handle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getAccessToken(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putAccessToken(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_accessToken(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getClientId(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putClientId(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_clientId(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getClientSecret(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putClientSecret(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_clientSecret(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getDebugLogFilePath(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putDebugLogFilePath(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_debugLogFilePath(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getLastErrorHtml(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_lastErrorHtml(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getLastErrorText(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_lastErrorText(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getLastErrorXml(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_lastErrorXml(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_getLastMethodSuccess(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putLastMethodSuccess(HCkAuthAzureAD cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAuthAzureAD_getNumSecondsRemaining(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getResource(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putResource(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_resource(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getTenantId(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putTenantId(HCkAuthAzureAD cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_tenantId(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_getUtf8(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putUtf8(HCkAuthAzureAD cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_getValid(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_getVerboseLogging(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_putVerboseLogging(HCkAuthAzureAD cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAzureAD_getVersion(HCkAuthAzureAD cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAzureAD_version(HCkAuthAzureAD cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_LoadTaskCaller(HCkAuthAzureAD cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_ObtainAccessToken(HCkAuthAzureAD cHandle, HCkSocket connection);
CK_C_VISIBLE_PUBLIC HCkTask CkAuthAzureAD_ObtainAccessTokenAsync(HCkAuthAzureAD cHandle, HCkSocket connection);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAzureAD_SaveLastError(HCkAuthAzureAD cHandle, const char *path);
#endif
