// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkGlobalWH
#define _C_CkGlobalWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkGlobalW CkGlobalW_Create(void);
CK_C_VISIBLE_PUBLIC void CkGlobalW_Dispose(HCkGlobalW handle);
CK_C_VISIBLE_PUBLIC int CkGlobalW_getAnsiCodePage(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putAnsiCodePage(HCkGlobalW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getDebugLogFilePath(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putDebugLogFilePath(HCkGlobalW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_debugLogFilePath(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC int CkGlobalW_getDefaultNtlmVersion(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putDefaultNtlmVersion(HCkGlobalW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getDefaultUtf8(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putDefaultUtf8(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkGlobalW_getDnsTimeToLive(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putDnsTimeToLive(HCkGlobalW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getEnableDnsCaching(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putEnableDnsCaching(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getLastErrorHtml(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_lastErrorHtml(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getLastErrorText(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_lastErrorText(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getLastErrorXml(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_lastErrorXml(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getLastMethodSuccess(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putLastMethodSuccess(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkGlobalW_getMaxThreads(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putMaxThreads(HCkGlobalW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getPreferIpv6(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putPreferIpv6(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getThreadPoolLogPath(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putThreadPoolLogPath(HCkGlobalW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_threadPoolLogPath(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC int CkGlobalW_getUnlockStatus(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getUsePkcsConstructedEncoding(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putUsePkcsConstructedEncoding(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getVerboseLogging(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putVerboseLogging(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_getVerboseTls(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC void  CkGlobalW_putVerboseTls(HCkGlobalW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkGlobalW_getVersion(HCkGlobalW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkGlobalW_version(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_DnsClearCache(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_FinalizeThreadPool(HCkGlobalW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_SaveLastError(HCkGlobalW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_ThreadPoolLogLine(HCkGlobalW cHandle, const wchar_t *str);
CK_C_VISIBLE_PUBLIC BOOL CkGlobalW_UnlockBundle(HCkGlobalW cHandle, const wchar_t *bundleUnlockCode);
#endif
