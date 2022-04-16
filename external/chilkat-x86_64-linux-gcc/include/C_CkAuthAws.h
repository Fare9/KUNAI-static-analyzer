// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAws_H
#define _C_CkAuthAws_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAuthAws CkAuthAws_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAws_Dispose(HCkAuthAws handle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getAccessKey(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putAccessKey(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_accessKey(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getCanonicalizedResourceV2(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putCanonicalizedResourceV2(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_canonicalizedResourceV2(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getDebugLogFilePath(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putDebugLogFilePath(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_debugLogFilePath(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getLastErrorHtml(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_lastErrorHtml(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getLastErrorText(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_lastErrorText(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getLastErrorXml(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_lastErrorXml(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAws_getLastMethodSuccess(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putLastMethodSuccess(HCkAuthAws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getPrecomputedMd5(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putPrecomputedMd5(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_precomputedMd5(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getPrecomputedSha256(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putPrecomputedSha256(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_precomputedSha256(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getRegion(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putRegion(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_region(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getSecretKey(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putSecretKey(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_secretKey(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getServiceName(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putServiceName(HCkAuthAws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_serviceName(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC int CkAuthAws_getSignatureVersion(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putSignatureVersion(HCkAuthAws cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAws_getUtf8(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putUtf8(HCkAuthAws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAws_getVerboseLogging(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAws_putVerboseLogging(HCkAuthAws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAws_getVersion(HCkAuthAws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_version(HCkAuthAws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAws_GenPresignedUrl(HCkAuthAws cHandle, const char *httpVerb, BOOL useHttps, const char *domain, const char *path, int numSecondsValid, const char *awsService, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAuthAws_genPresignedUrl(HCkAuthAws cHandle, const char *httpVerb, BOOL useHttps, const char *domain, const char *path, int numSecondsValid, const char *awsService);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAws_SaveLastError(HCkAuthAws cHandle, const char *path);
#endif
