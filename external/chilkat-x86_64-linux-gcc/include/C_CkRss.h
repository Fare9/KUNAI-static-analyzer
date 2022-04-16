// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkRss_H
#define _C_CkRss_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkRss_setAbortCheck(HCkRss cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkRss_setPercentDone(HCkRss cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkRss_setProgressInfo(HCkRss cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkRss_setTaskCompleted(HCkRss cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkRss_setAbortCheck2(HCkRss cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkRss_setPercentDone2(HCkRss cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkRss_setProgressInfo2(HCkRss cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkRss_setTaskCompleted2(HCkRss cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkRss_setExternalProgress(HCkRss cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkRss_setCallbackContext(HCkRss cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkRss CkRss_Create(void);
CK_C_VISIBLE_PUBLIC void CkRss_Dispose(HCkRss handle);
CK_C_VISIBLE_PUBLIC void CkRss_getDebugLogFilePath(HCkRss cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkRss_putDebugLogFilePath(HCkRss cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkRss_debugLogFilePath(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_getLastErrorHtml(HCkRss cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkRss_lastErrorHtml(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_getLastErrorText(HCkRss cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkRss_lastErrorText(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_getLastErrorXml(HCkRss cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkRss_lastErrorXml(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRss_getLastMethodSuccess(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_putLastMethodSuccess(HCkRss cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkRss_getNumChannels(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC int CkRss_getNumItems(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRss_getUtf8(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_putUtf8(HCkRss cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkRss_getVerboseLogging(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_putVerboseLogging(HCkRss cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkRss_getVersion(HCkRss cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkRss_version(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_AddNewChannel(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_AddNewImage(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_AddNewItem(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkRss_DownloadRss(HCkRss cHandle, const char *url);
CK_C_VISIBLE_PUBLIC HCkTask CkRss_DownloadRssAsync(HCkRss cHandle, const char *url);
CK_C_VISIBLE_PUBLIC BOOL CkRss_GetAttr(HCkRss cHandle, const char *tag, const char *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_getAttr(HCkRss cHandle, const char *tag, const char *attrName);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_GetChannel(HCkRss cHandle, int index);
CK_C_VISIBLE_PUBLIC int CkRss_GetCount(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRss_GetDate(HCkRss cHandle, const char *tag, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC BOOL CkRss_GetDateStr(HCkRss cHandle, const char *tag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_getDateStr(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_GetImage(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC int CkRss_GetInt(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC HCkRss CkRss_GetItem(HCkRss cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkRss_GetString(HCkRss cHandle, const char *tag, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_getString(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRss_LoadRssFile(HCkRss cHandle, const char *filePath);
CK_C_VISIBLE_PUBLIC BOOL CkRss_LoadRssString(HCkRss cHandle, const char *rssString);
CK_C_VISIBLE_PUBLIC BOOL CkRss_LoadTaskCaller(HCkRss cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkRss_MGetAttr(HCkRss cHandle, const char *tag, int index, const char *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_mGetAttr(HCkRss cHandle, const char *tag, int index, const char *attrName);
CK_C_VISIBLE_PUBLIC BOOL CkRss_MGetString(HCkRss cHandle, const char *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_mGetString(HCkRss cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkRss_MSetAttr(HCkRss cHandle, const char *tag, int idx, const char *attrName, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkRss_MSetString(HCkRss cHandle, const char *tag, int idx, const char *value);
CK_C_VISIBLE_PUBLIC void CkRss_NewRss(HCkRss cHandle);
CK_C_VISIBLE_PUBLIC void CkRss_Remove(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC BOOL CkRss_SaveLastError(HCkRss cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkRss_SetAttr(HCkRss cHandle, const char *tag, const char *attrName, const char *value);
CK_C_VISIBLE_PUBLIC void CkRss_SetDate(HCkRss cHandle, const char *tag, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC void CkRss_SetDateNow(HCkRss cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC void CkRss_SetDateStr(HCkRss cHandle, const char *tag, const char *dateTimeStr);
CK_C_VISIBLE_PUBLIC void CkRss_SetInt(HCkRss cHandle, const char *tag, int value);
CK_C_VISIBLE_PUBLIC void CkRss_SetString(HCkRss cHandle, const char *tag, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkRss_ToXmlString(HCkRss cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkRss_toXmlString(HCkRss cHandle);
#endif
