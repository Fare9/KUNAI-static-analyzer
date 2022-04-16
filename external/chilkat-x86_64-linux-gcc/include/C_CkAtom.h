// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAtom_H
#define _C_CkAtom_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAtom_setAbortCheck(HCkAtom cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAtom_setPercentDone(HCkAtom cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAtom_setProgressInfo(HCkAtom cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkAtom_setTaskCompleted(HCkAtom cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkAtom_setAbortCheck2(HCkAtom cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkAtom_setPercentDone2(HCkAtom cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAtom_setProgressInfo2(HCkAtom cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkAtom_setTaskCompleted2(HCkAtom cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkAtom_setExternalProgress(HCkAtom cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkAtom_setCallbackContext(HCkAtom cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkAtom CkAtom_Create(void);
CK_C_VISIBLE_PUBLIC void CkAtom_Dispose(HCkAtom handle);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_getAbortCurrent(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_getDebugLogFilePath(HCkAtom cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAtom_putDebugLogFilePath(HCkAtom cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAtom_debugLogFilePath(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_getLastErrorHtml(HCkAtom cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAtom_lastErrorHtml(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_getLastErrorText(HCkAtom cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAtom_lastErrorText(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_getLastErrorXml(HCkAtom cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAtom_lastErrorXml(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_getLastMethodSuccess(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_putLastMethodSuccess(HCkAtom cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAtom_getNumEntries(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_getUtf8(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_putUtf8(HCkAtom cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_getVerboseLogging(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_putVerboseLogging(HCkAtom cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAtom_getVersion(HCkAtom cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAtom_version(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElement(HCkAtom cHandle, const char *tag, const char *value);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementDate(HCkAtom cHandle, const char *tag, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementDateStr(HCkAtom cHandle, const char *tag, const char *dateTimeStr);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementDt(HCkAtom cHandle, const char *tag, HCkDateTime dateTime);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementHtml(HCkAtom cHandle, const char *tag, const char *htmlStr);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementXHtml(HCkAtom cHandle, const char *tag, const char *xmlStr);
CK_C_VISIBLE_PUBLIC int CkAtom_AddElementXml(HCkAtom cHandle, const char *tag, const char *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_AddEntry(HCkAtom cHandle, const char *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_AddLink(HCkAtom cHandle, const char *rel, const char *href, const char *title, const char *typ);
CK_C_VISIBLE_PUBLIC void CkAtom_AddPerson(HCkAtom cHandle, const char *tag, const char *name, const char *uri, const char *email);
CK_C_VISIBLE_PUBLIC void CkAtom_DeleteElement(HCkAtom cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC void CkAtom_DeleteElementAttr(HCkAtom cHandle, const char *tag, int index, const char *attrName);
CK_C_VISIBLE_PUBLIC void CkAtom_DeletePerson(HCkAtom cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_DownloadAtom(HCkAtom cHandle, const char *url);
CK_C_VISIBLE_PUBLIC HCkTask CkAtom_DownloadAtomAsync(HCkAtom cHandle, const char *url);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetElement(HCkAtom cHandle, const char *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getElement(HCkAtom cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetElementAttr(HCkAtom cHandle, const char *tag, int index, const char *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getElementAttr(HCkAtom cHandle, const char *tag, int index, const char *attrName);
CK_C_VISIBLE_PUBLIC int CkAtom_GetElementCount(HCkAtom cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetElementDate(HCkAtom cHandle, const char *tag, int index, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetElementDateStr(HCkAtom cHandle, const char *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getElementDateStr(HCkAtom cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC HCkDateTime CkAtom_GetElementDt(HCkAtom cHandle, const char *tag, int index);
CK_C_VISIBLE_PUBLIC HCkAtom CkAtom_GetEntry(HCkAtom cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetLinkHref(HCkAtom cHandle, const char *relName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getLinkHref(HCkAtom cHandle, const char *relName);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetPersonInfo(HCkAtom cHandle, const char *tag, int index, const char *tag2, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getPersonInfo(HCkAtom cHandle, const char *tag, int index, const char *tag2);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_GetTopAttr(HCkAtom cHandle, const char *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_getTopAttr(HCkAtom cHandle, const char *attrName);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_HasElement(HCkAtom cHandle, const char *tag);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_LoadTaskCaller(HCkAtom cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_LoadXml(HCkAtom cHandle, const char *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_NewEntry(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_NewFeed(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_SaveLastError(HCkAtom cHandle, const char *path);
CK_C_VISIBLE_PUBLIC void CkAtom_SetElementAttr(HCkAtom cHandle, const char *tag, int index, const char *attrName, const char *attrValue);
CK_C_VISIBLE_PUBLIC void CkAtom_SetTopAttr(HCkAtom cHandle, const char *attrName, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkAtom_ToXmlString(HCkAtom cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAtom_toXmlString(HCkAtom cHandle);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElement(HCkAtom cHandle, const char *tag, int index, const char *value);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementDate(HCkAtom cHandle, const char *tag, int index, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementDateStr(HCkAtom cHandle, const char *tag, int index, const char *dateTimeStr);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementDt(HCkAtom cHandle, const char *tag, int index, HCkDateTime dateTime);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementHtml(HCkAtom cHandle, const char *tag, int index, const char *htmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementXHtml(HCkAtom cHandle, const char *tag, int index, const char *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdateElementXml(HCkAtom cHandle, const char *tag, int index, const char *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtom_UpdatePerson(HCkAtom cHandle, const char *tag, int index, const char *name, const char *uri, const char *email);
#endif
