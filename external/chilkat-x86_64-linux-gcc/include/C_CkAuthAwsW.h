// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAuthAwsWH
#define _C_CkAuthAwsWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAuthAwsW CkAuthAwsW_Create(void);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_Dispose(HCkAuthAwsW handle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getAccessKey(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putAccessKey(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_accessKey(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getCanonicalizedResourceV2(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putCanonicalizedResourceV2(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_canonicalizedResourceV2(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getDebugLogFilePath(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putDebugLogFilePath(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_debugLogFilePath(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getLastErrorHtml(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_lastErrorHtml(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getLastErrorText(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_lastErrorText(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getLastErrorXml(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_lastErrorXml(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAwsW_getLastMethodSuccess(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putLastMethodSuccess(HCkAuthAwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getPrecomputedMd5(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putPrecomputedMd5(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_precomputedMd5(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getPrecomputedSha256(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putPrecomputedSha256(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_precomputedSha256(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getRegion(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putRegion(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_region(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getSecretKey(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putSecretKey(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_secretKey(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getServiceName(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putServiceName(HCkAuthAwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_serviceName(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC int CkAuthAwsW_getSignatureVersion(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putSignatureVersion(HCkAuthAwsW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAwsW_getVerboseLogging(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAuthAwsW_putVerboseLogging(HCkAuthAwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAuthAwsW_getVersion(HCkAuthAwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_version(HCkAuthAwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAwsW_GenPresignedUrl(HCkAuthAwsW cHandle, const wchar_t *httpVerb, BOOL useHttps, const wchar_t *domain, const wchar_t *path, int numSecondsValid, const wchar_t *awsService, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAuthAwsW_genPresignedUrl(HCkAuthAwsW cHandle, const wchar_t *httpVerb, BOOL useHttps, const wchar_t *domain, const wchar_t *path, int numSecondsValid, const wchar_t *awsService);
CK_C_VISIBLE_PUBLIC BOOL CkAuthAwsW_SaveLastError(HCkAuthAwsW cHandle, const wchar_t *path);
#endif
