// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAsn_H
#define _C_CkAsn_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAsn CkAsn_Create(void);
CK_C_VISIBLE_PUBLIC void CkAsn_Dispose(HCkAsn handle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_getBoolValue(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_putBoolValue(HCkAsn cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_getConstructed(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_getContentStr(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAsn_putContentStr(HCkAsn cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAsn_contentStr(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_getDebugLogFilePath(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkAsn_putDebugLogFilePath(HCkAsn cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkAsn_debugLogFilePath(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC int CkAsn_getIntValue(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_putIntValue(HCkAsn cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkAsn_getLastErrorHtml(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAsn_lastErrorHtml(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_getLastErrorText(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAsn_lastErrorText(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_getLastErrorXml(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAsn_lastErrorXml(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_getLastMethodSuccess(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_putLastMethodSuccess(HCkAsn cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAsn_getNumSubItems(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_getTag(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAsn_tag(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC int CkAsn_getTagValue(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_getUtf8(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_putUtf8(HCkAsn cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_getVerboseLogging(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC void CkAsn_putVerboseLogging(HCkAsn cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAsn_getVersion(HCkAsn cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkAsn_version(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendBigInt(HCkAsn cHandle, const char *encodedBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendBits(HCkAsn cHandle, const char *encodedBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendBool(HCkAsn cHandle, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendContextConstructed(HCkAsn cHandle, int tag);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendContextPrimitive(HCkAsn cHandle, int tag, const char *encodedBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendInt(HCkAsn cHandle, int value);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendNull(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendOctets(HCkAsn cHandle, const char *encodedBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendOid(HCkAsn cHandle, const char *oid);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendSequence(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendSequence2(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC HCkAsn CkAsn_AppendSequenceR(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendSet(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendSet2(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC HCkAsn CkAsn_AppendSetR(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendString(HCkAsn cHandle, const char *strType, const char *value);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AppendTime(HCkAsn cHandle, const char *timeFormat, const char *dateTimeStr);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_AsnToXml(HCkAsn cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAsn_asnToXml(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_DeleteSubItem(HCkAsn cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_GetBinaryDer(HCkAsn cHandle, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_GetEncodedContent(HCkAsn cHandle, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAsn_getEncodedContent(HCkAsn cHandle, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_GetEncodedDer(HCkAsn cHandle, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkAsn_getEncodedDer(HCkAsn cHandle, const char *encoding);
CK_C_VISIBLE_PUBLIC HCkAsn CkAsn_GetLastSubItem(HCkAsn cHandle);
CK_C_VISIBLE_PUBLIC HCkAsn CkAsn_GetSubItem(HCkAsn cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_LoadAsnXml(HCkAsn cHandle, const char *xmlStr);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_LoadBd(HCkAsn cHandle, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_LoadBinary(HCkAsn cHandle, HCkByteData derBytes);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_LoadBinaryFile(HCkAsn cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_LoadEncoded(HCkAsn cHandle, const char *asnContent, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_SaveLastError(HCkAsn cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_SetEncodedContent(HCkAsn cHandle, const char *encodedBytes, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_WriteBd(HCkAsn cHandle, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkAsn_WriteBinaryDer(HCkAsn cHandle, const char *path);
#endif
