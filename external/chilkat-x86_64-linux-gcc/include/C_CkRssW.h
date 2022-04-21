// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkRssWH
#define _C_CkRssWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkRssW_setAbortCheck(HCkRssW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkRssW_setPercentDone(HCkRssW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkRssW_setProgressInfo(HCkRssW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkRssW_setTaskCompleted(HCkRssW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_Create(void);
CK_C_VISIBLE_PUBLIC void CkRssW_Dispose(HCkRssW handle);
CK_C_VISIBLE_PUBLIC void CkRssW_getDebugLogFilePath(HCkRssW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkRssW_putDebugLogFilePath(HCkRssW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_debugLogFilePath(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void CkRssW_getLastErrorHtml(HCkRssW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_lastErrorHtml(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void CkRssW_getLastErrorText(HCkRssW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_lastErrorText(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void CkRssW_getLastErrorXml(HCkRssW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_lastErrorXml(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_getLastMethodSuccess(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void  CkRssW_putLastMethodSuccess(HCkRssW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkRssW_getNumChannels(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC int CkRssW_getNumItems(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_getVerboseLogging(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void  CkRssW_putVerboseLogging(HCkRssW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkRssW_getVersion(HCkRssW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_version(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_AddNewChannel(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_AddNewImage(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_AddNewItem(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_DownloadRss(HCkRssW cHandle, const wchar_t *url);
CK_C_VISIBLE_PUBLIC HCkTaskW CkRssW_DownloadRssAsync(HCkRssW cHandle, const wchar_t *url);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_GetAttr(HCkRssW cHandle, const wchar_t *tag, const wchar_t *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_getAttr(HCkRssW cHandle, const wchar_t *tag, const wchar_t *attrName);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_GetChannel(HCkRssW cHandle, int index);
CK_C_VISIBLE_PUBLIC int CkRssW_GetCount(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_GetDate(HCkRssW cHandle, const wchar_t *tag, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_GetDateStr(HCkRssW cHandle, const wchar_t *tag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_getDateStr(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_GetImage(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC int CkRssW_GetInt(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC HCkRssW CkRssW_GetItem(HCkRssW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_GetString(HCkRssW cHandle, const wchar_t *tag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_getString(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_LoadRssFile(HCkRssW cHandle, const wchar_t *filePath);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_LoadRssString(HCkRssW cHandle, const wchar_t *rssString);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_LoadTaskCaller(HCkRssW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_MGetAttr(HCkRssW cHandle, const wchar_t *tag, int index, const wchar_t *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_mGetAttr(HCkRssW cHandle, const wchar_t *tag, int index, const wchar_t *attrName);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_MGetString(HCkRssW cHandle, const wchar_t *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_mGetString(HCkRssW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_MSetAttr(HCkRssW cHandle, const wchar_t *tag, int idx, const wchar_t *attrName, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_MSetString(HCkRssW cHandle, const wchar_t *tag, int idx, const wchar_t *value);
CK_C_VISIBLE_PUBLIC void CkRssW_NewRss(HCkRssW cHandle);
CK_C_VISIBLE_PUBLIC void CkRssW_Remove(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_SaveLastError(HCkRssW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC void CkRssW_SetAttr(HCkRssW cHandle, const wchar_t *tag, const wchar_t *attrName, const wchar_t *value);
CK_C_VISIBLE_PUBLIC void CkRssW_SetDate(HCkRssW cHandle, const wchar_t *tag, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC void CkRssW_SetDateNow(HCkRssW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC void CkRssW_SetDateStr(HCkRssW cHandle, const wchar_t *tag, const wchar_t *dateTimeStr);
CK_C_VISIBLE_PUBLIC void CkRssW_SetInt(HCkRssW cHandle, const wchar_t *tag, int value);
CK_C_VISIBLE_PUBLIC void CkRssW_SetString(HCkRssW cHandle, const wchar_t *tag, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkRssW_ToXmlString(HCkRssW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkRssW_toXmlString(HCkRssW cHandle);
#endif
