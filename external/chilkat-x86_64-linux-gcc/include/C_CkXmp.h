// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkXmp_H
#define _C_CkXmp_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkXmp CkXmp_Create(void);
CK_C_VISIBLE_PUBLIC void CkXmp_Dispose(HCkXmp handle);
CK_C_VISIBLE_PUBLIC void CkXmp_getDebugLogFilePath(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkXmp_putDebugLogFilePath(HCkXmp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkXmp_debugLogFilePath(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_getLastErrorHtml(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmp_lastErrorHtml(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_getLastErrorText(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmp_lastErrorText(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_getLastErrorXml(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmp_lastErrorXml(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_getLastMethodSuccess(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_putLastMethodSuccess(HCkXmp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkXmp_getNumEmbedded(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_getStructInnerDescrip(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_putStructInnerDescrip(HCkXmp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmp_getUncommonOptions(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkXmp_putUncommonOptions(HCkXmp cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkXmp_uncommonOptions(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_getUtf8(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_putUtf8(HCkXmp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_getVerboseLogging(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC void CkXmp_putVerboseLogging(HCkXmp cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkXmp_getVersion(HCkXmp cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkXmp_version(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_AddArray(HCkXmp cHandle, HCkXml xml, const char *arrType, const char *propName, HCkStringArray values);
CK_C_VISIBLE_PUBLIC void CkXmp_AddNsMapping(HCkXmp cHandle, const char *ns, const char *uri);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_AddSimpleDate(HCkXmp cHandle, HCkXml iXml, const char *propName, SYSTEMTIME * propVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_AddSimpleInt(HCkXmp cHandle, HCkXml iXml, const char *propName, int propVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_AddSimpleStr(HCkXmp cHandle, HCkXml iXml, const char *propName, const char *propVal);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_AddStructProp(HCkXmp cHandle, HCkXml iChilkatXml, const char *structName, const char *propName, const char *propValue);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_Append(HCkXmp cHandle, HCkXml iXml);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_DateToString(HCkXmp cHandle, SYSTEMTIME * d, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmp_dateToString(HCkXmp cHandle, SYSTEMTIME * d);
CK_C_VISIBLE_PUBLIC HCkStringArray CkXmp_GetArray(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC HCkXml CkXmp_GetEmbedded(HCkXmp cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkXml CkXmp_GetProperty(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_GetSimpleDate(HCkXmp cHandle, HCkXml iXml, const char *propName, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC int CkXmp_GetSimpleInt(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_GetSimpleStr(HCkXmp cHandle, HCkXml iXml, const char *propName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmp_getSimpleStr(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC HCkStringArray CkXmp_GetStructPropNames(HCkXmp cHandle, HCkXml iXml, const char *structName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_GetStructValue(HCkXmp cHandle, HCkXml iXml, const char *structName, const char *propName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkXmp_getStructValue(HCkXmp cHandle, HCkXml iXml, const char *structName, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_LoadAppFile(HCkXmp cHandle, const char *filename);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_LoadFromBuffer(HCkXmp cHandle, HCkByteData fileData, const char *ext);
CK_C_VISIBLE_PUBLIC HCkXml CkXmp_NewXmp(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveAllEmbedded(HCkXmp cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveArray(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveEmbedded(HCkXmp cHandle, int index);
CK_C_VISIBLE_PUBLIC void CkXmp_RemoveNsMapping(HCkXmp cHandle, const char *ns);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveSimple(HCkXmp cHandle, HCkXml iXml, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveStruct(HCkXmp cHandle, HCkXml iXml, const char *structName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_RemoveStructProp(HCkXmp cHandle, HCkXml iXml, const char *structName, const char *propName);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_SaveAppFile(HCkXmp cHandle, const char *filename);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_SaveLastError(HCkXmp cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_SaveToBuffer(HCkXmp cHandle, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_StringToDate(HCkXmp cHandle, const char *str, SYSTEMTIME *outSysTime);
CK_C_VISIBLE_PUBLIC BOOL CkXmp_UnlockComponent(HCkXmp cHandle, const char *unlockCode);
#endif
