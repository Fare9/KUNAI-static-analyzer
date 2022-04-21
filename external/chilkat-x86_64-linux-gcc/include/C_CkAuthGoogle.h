// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthGoogle_H
#define _C_CkAuthGoogle_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setAbortCheck(HCkAuthGoogle cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setPercentDone(HCkAuthGoogle cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setProgressInfo(HCkAuthGoogle cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setTaskCompleted(HCkAuthGoogle cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setAbortCheck2(HCkAuthGoogle cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setPercentDone2(HCkAuthGoogle cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setProgressInfo2(HCkAuthGoogle cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setTaskCompleted2(HCkAuthGoogle cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setExternalProgress(HCkAuthGoogle cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_setCallbackContext(HCkAuthGoogle cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkAuthGoogle CkAuthGoogle_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_Dispose(HCkAuthGoogle handle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getAccessToken(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putAccessToken(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_accessToken(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getDebugLogFilePath(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putDebugLogFilePath(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_debugLogFilePath(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getEmailAddress(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putEmailAddress(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_emailAddress(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC int CkAuthGoogle_getExpireNumSeconds(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putExpireNumSeconds(HCkAuthGoogle cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkAuthGoogle_getIat(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putIat(HCkAuthGoogle cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getJsonKey(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putJsonKey(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_jsonKey(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getLastErrorHtml(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_lastErrorHtml(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getLastErrorText(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_lastErrorText(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getLastErrorXml(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_lastErrorXml(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_getLastMethodSuccess(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putLastMethodSuccess(HCkAuthGoogle cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAuthGoogle_getNumSecondsRemaining(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getScope(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putScope(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_scope(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getSubEmailAddress(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putSubEmailAddress(HCkAuthGoogle cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_subEmailAddress(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_getUtf8(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putUtf8(HCkAuthGoogle cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_getValid(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_getVerboseLogging(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_putVerboseLogging(HCkAuthGoogle cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthGoogle_getVersion(HCkAuthGoogle cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthGoogle_version(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC HCkPfx CkAuthGoogle_GetP12(HCkAuthGoogle cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_LoadTaskCaller(HCkAuthGoogle cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_ObtainAccessToken(HCkAuthGoogle cHandle, HCkSocket connection);
CK_C_VISIBLE_PUBLIC HCkTask CkAuthGoogle_ObtainAccessTokenAsync(HCkAuthGoogle cHandle, HCkSocket connection);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_SaveLastError(HCkAuthGoogle cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogle_SetP12(HCkAuthGoogle cHandle, HCkPfx key);
#endif
