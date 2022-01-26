// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAtomWH
#define _C_CkAtomWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkAtomW_setAbortCheck(HCkAtomW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkAtomW_setPercentDone(HCkAtomW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkAtomW_setProgressInfo(HCkAtomW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkAtomW_setTaskCompleted(HCkAtomW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkAtomW CkAtomW_Create(void);
CK_C_VISIBLE_PUBLIC void CkAtomW_Dispose(HCkAtomW handle);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_getAbortCurrent(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_getDebugLogFilePath(HCkAtomW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAtomW_putDebugLogFilePath(HCkAtomW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_debugLogFilePath(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_getLastErrorHtml(HCkAtomW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_lastErrorHtml(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_getLastErrorText(HCkAtomW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_lastErrorText(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_getLastErrorXml(HCkAtomW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_lastErrorXml(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_getLastMethodSuccess(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAtomW_putLastMethodSuccess(HCkAtomW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAtomW_getNumEntries(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_getVerboseLogging(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAtomW_putVerboseLogging(HCkAtomW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAtomW_getVersion(HCkAtomW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_version(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElement(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *value);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementDate(HCkAtomW cHandle, const wchar_t *tag, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementDateStr(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *dateTimeStr);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementDt(HCkAtomW cHandle, const wchar_t *tag, HCkDateTimeW dateTime);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementHtml(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *htmlStr);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementXHtml(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC int CkAtomW_AddElementXml(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_AddEntry(HCkAtomW cHandle, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_AddLink(HCkAtomW cHandle, const wchar_t *rel, const wchar_t *href, const wchar_t *title, const wchar_t *typ);
CK_C_VISIBLE_PUBLIC void CkAtomW_AddPerson(HCkAtomW cHandle, const wchar_t *tag, const wchar_t *name, const wchar_t *uri, const wchar_t *email);
CK_C_VISIBLE_PUBLIC void CkAtomW_DeleteElement(HCkAtomW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC void CkAtomW_DeleteElementAttr(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *attrName);
CK_C_VISIBLE_PUBLIC void CkAtomW_DeletePerson(HCkAtomW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_DownloadAtom(HCkAtomW cHandle, const wchar_t *url);
CK_C_VISIBLE_PUBLIC HCkTaskW CkAtomW_DownloadAtomAsync(HCkAtomW cHandle, const wchar_t *url);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetElement(HCkAtomW cHandle, const wchar_t *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getElement(HCkAtomW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetElementAttr(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getElementAttr(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *attrName);
CK_C_VISIBLE_PUBLIC int CkAtomW_GetElementCount(HCkAtomW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetElementDate(HCkAtomW cHandle, const wchar_t *tag, int index, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetElementDateStr(HCkAtomW cHandle, const wchar_t *tag, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getElementDateStr(HCkAtomW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC HCkDateTimeW CkAtomW_GetElementDt(HCkAtomW cHandle, const wchar_t *tag, int index);
CK_C_VISIBLE_PUBLIC HCkAtomW CkAtomW_GetEntry(HCkAtomW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetLinkHref(HCkAtomW cHandle, const wchar_t *relName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getLinkHref(HCkAtomW cHandle, const wchar_t *relName);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetPersonInfo(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *tag2, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getPersonInfo(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *tag2);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_GetTopAttr(HCkAtomW cHandle, const wchar_t *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_getTopAttr(HCkAtomW cHandle, const wchar_t *attrName);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_HasElement(HCkAtomW cHandle, const wchar_t *tag);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_LoadTaskCaller(HCkAtomW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_LoadXml(HCkAtomW cHandle, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_NewEntry(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_NewFeed(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_SaveLastError(HCkAtomW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC void CkAtomW_SetElementAttr(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *attrName, const wchar_t *attrValue);
CK_C_VISIBLE_PUBLIC void CkAtomW_SetTopAttr(HCkAtomW cHandle, const wchar_t *attrName, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkAtomW_ToXmlString(HCkAtomW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAtomW_toXmlString(HCkAtomW cHandle);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElement(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *value);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementDate(HCkAtomW cHandle, const wchar_t *tag, int index, SYSTEMTIME * dateTime);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementDateStr(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *dateTimeStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementDt(HCkAtomW cHandle, const wchar_t *tag, int index, HCkDateTimeW dateTime);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementHtml(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *htmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementXHtml(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdateElementXml(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC void CkAtomW_UpdatePerson(HCkAtomW cHandle, const wchar_t *tag, int index, const wchar_t *name, const wchar_t *uri, const wchar_t *email);
#endif
