// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkEmailBundleWH
#define _C_CkEmailBundleWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkEmailBundleW CkEmailBundleW_Create(void);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_Dispose(HCkEmailBundleW handle);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_getDebugLogFilePath(HCkEmailBundleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkEmailBundleW_putDebugLogFilePath(HCkEmailBundleW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_debugLogFilePath(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_getLastErrorHtml(HCkEmailBundleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_lastErrorHtml(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_getLastErrorText(HCkEmailBundleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_lastErrorText(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_getLastErrorXml(HCkEmailBundleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_lastErrorXml(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_getLastMethodSuccess(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkEmailBundleW_putLastMethodSuccess(HCkEmailBundleW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkEmailBundleW_getMessageCount(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_getVerboseLogging(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC void  CkEmailBundleW_putVerboseLogging(HCkEmailBundleW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_getVersion(HCkEmailBundleW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_version(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_AddEmail(HCkEmailBundleW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC HCkEmailW CkEmailBundleW_FindByHeader(HCkEmailBundleW cHandle, const wchar_t *headerFieldName, const wchar_t *headerFieldValue);
CK_C_VISIBLE_PUBLIC HCkEmailW CkEmailBundleW_GetEmail(HCkEmailBundleW cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkStringArrayW CkEmailBundleW_GetUidls(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_GetXml(HCkEmailBundleW cHandle, HCkString outXml);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEmailBundleW_getXml(HCkEmailBundleW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_LoadTaskResult(HCkEmailBundleW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_LoadXml(HCkEmailBundleW cHandle, const wchar_t *filename);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_LoadXmlString(HCkEmailBundleW cHandle, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_RemoveEmail(HCkEmailBundleW cHandle, HCkEmailW email);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_RemoveEmailByIndex(HCkEmailBundleW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_SaveLastError(HCkEmailBundleW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkEmailBundleW_SaveXml(HCkEmailBundleW cHandle, const wchar_t *filename);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_SortByDate(HCkEmailBundleW cHandle, BOOL ascending);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_SortByRecipient(HCkEmailBundleW cHandle, BOOL ascending);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_SortBySender(HCkEmailBundleW cHandle, BOOL ascending);
CK_C_VISIBLE_PUBLIC void CkEmailBundleW_SortBySubject(HCkEmailBundleW cHandle, BOOL ascending);
#endif
