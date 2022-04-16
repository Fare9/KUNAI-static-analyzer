// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthGoogleWH
#define _C_CkAuthGoogleWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_setAbortCheck(HCkAuthGoogleW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_setPercentDone(HCkAuthGoogleW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_setProgressInfo(HCkAuthGoogleW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_setTaskCompleted(HCkAuthGoogleW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkAuthGoogleW CkAuthGoogleW_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_Dispose(HCkAuthGoogleW handle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getAccessToken(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putAccessToken(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_accessToken(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getDebugLogFilePath(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putDebugLogFilePath(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_debugLogFilePath(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getEmailAddress(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putEmailAddress(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_emailAddress(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC int CkAuthGoogleW_getExpireNumSeconds(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putExpireNumSeconds(HCkAuthGoogleW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkAuthGoogleW_getIat(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putIat(HCkAuthGoogleW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getJsonKey(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putJsonKey(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_jsonKey(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getLastErrorHtml(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_lastErrorHtml(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getLastErrorText(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_lastErrorText(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getLastErrorXml(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_lastErrorXml(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_getLastMethodSuccess(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putLastMethodSuccess(HCkAuthGoogleW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAuthGoogleW_getNumSecondsRemaining(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getScope(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putScope(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_scope(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getSubEmailAddress(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putSubEmailAddress(HCkAuthGoogleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_subEmailAddress(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_getValid(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_getVerboseLogging(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthGoogleW_putVerboseLogging(HCkAuthGoogleW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthGoogleW_getVersion(HCkAuthGoogleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthGoogleW_version(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC HCkPfxW CkAuthGoogleW_GetP12(HCkAuthGoogleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_LoadTaskCaller(HCkAuthGoogleW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_ObtainAccessToken(HCkAuthGoogleW cHandle, HCkSocketW connection);
CK_C_VISIBLE_PUBLIC HCkTaskW CkAuthGoogleW_ObtainAccessTokenAsync(HCkAuthGoogleW cHandle, HCkSocketW connection);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_SaveLastError(HCkAuthGoogleW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkAuthGoogleW_SetP12(HCkAuthGoogleW cHandle, HCkPfxW key);
#endif
