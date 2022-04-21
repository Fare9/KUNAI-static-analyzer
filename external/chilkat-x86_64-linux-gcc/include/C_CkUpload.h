// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkUpload_H
#define _C_CkUpload_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkUpload_setAbortCheck(HCkUpload cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkUpload_setPercentDone(HCkUpload cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkUpload_setProgressInfo(HCkUpload cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkUpload_setTaskCompleted(HCkUpload cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkUpload_setAbortCheck2(HCkUpload cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkUpload_setPercentDone2(HCkUpload cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkUpload_setProgressInfo2(HCkUpload cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkUpload_setTaskCompleted2(HCkUpload cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkUpload_setExternalProgress(HCkUpload cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkUpload_setCallbackContext(HCkUpload cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkUpload CkUpload_Create(void);
CK_C_VISIBLE_PUBLIC void CkUpload_Dispose(HCkUpload handle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getAbortCurrent(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putAbortCurrent(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkUpload_getBandwidthThrottleUp(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putBandwidthThrottleUp(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC int CkUpload_getChunkSize(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putChunkSize(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getClientIpAddress(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putClientIpAddress(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_clientIpAddress(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getDebugLogFilePath(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putDebugLogFilePath(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_debugLogFilePath(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getExpect100Continue(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putExpect100Continue(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkUpload_getHeartbeatMs(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putHeartbeatMs(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getHostname(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putHostname(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_hostname(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC int CkUpload_getIdleTimeoutMs(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putIdleTimeoutMs(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getLastErrorHtml(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_lastErrorHtml(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getLastErrorText(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_lastErrorText(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getLastErrorXml(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_lastErrorXml(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getLastMethodSuccess(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putLastMethodSuccess(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getLogin(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putLogin(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_login(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkUpload_getNumBytesSent(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getPassword(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putPassword(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_password(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getPath(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putPath(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_path(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC int CkUpload_getPercentDoneScale(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putPercentDoneScale(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC unsigned long CkUpload_getPercentUploaded(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC int CkUpload_getPort(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putPort(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getPreferIpv6(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putPreferIpv6(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getProxyDomain(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putProxyDomain(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_proxyDomain(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getProxyLogin(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putProxyLogin(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_proxyLogin(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getProxyPassword(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putProxyPassword(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_proxyPassword(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC int CkUpload_getProxyPort(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putProxyPort(HCkUpload cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getResponseBody(HCkUpload cHandle, HCkByteData retval);
CK_C_VISIBLE_PUBLIC void CkUpload_getResponseBodyStr(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_responseBodyStr(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getResponseHeader(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_responseHeader(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC int CkUpload_getResponseStatus(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getSsl(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putSsl(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getSslAllowedCiphers(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putSslAllowedCiphers(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_sslAllowedCiphers(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getSslProtocol(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putSslProtocol(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_sslProtocol(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_getTlsPinSet(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkUpload_putTlsPinSet(HCkUpload cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkUpload_tlsPinSet(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC unsigned long CkUpload_getTotalUploadSize(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getUploadInProgress(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getUploadSuccess(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getUtf8(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putUtf8(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_getVerboseLogging(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_putVerboseLogging(HCkUpload cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUpload_getVersion(HCkUpload cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUpload_version(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_AbortUpload(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_AddCustomHeader(HCkUpload cHandle, const char *name, const char *value);
CK_C_VISIBLE_PUBLIC void CkUpload_AddFileReference(HCkUpload cHandle, const char *name, const char *filename);
CK_C_VISIBLE_PUBLIC void CkUpload_AddParam(HCkUpload cHandle, const char *name, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_BeginUpload(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_BlockingUpload(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC HCkTask CkUpload_BlockingUploadAsync(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_ClearFileReferences(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC void CkUpload_ClearParams(HCkUpload cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_LoadTaskCaller(HCkUpload cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_SaveLastError(HCkUpload cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkUpload_SleepMs(HCkUpload cHandle, int millisec);
CK_C_VISIBLE_PUBLIC BOOL CkUpload_UploadToMemory(HCkUpload cHandle, HCkByteData outData);
#endif
