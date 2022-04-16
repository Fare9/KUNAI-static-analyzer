// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkAsnWH
#define _C_CkAsnWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkAsnW CkAsnW_Create(void);
CK_C_VISIBLE_PUBLIC void CkAsnW_Dispose(HCkAsnW handle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_getBoolValue(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putBoolValue(HCkAsnW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_getConstructed(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void CkAsnW_getContentStr(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putContentStr(HCkAsnW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_contentStr(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void CkAsnW_getDebugLogFilePath(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putDebugLogFilePath(HCkAsnW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_debugLogFilePath(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC int CkAsnW_getIntValue(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putIntValue(HCkAsnW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkAsnW_getLastErrorHtml(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_lastErrorHtml(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void CkAsnW_getLastErrorText(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_lastErrorText(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void CkAsnW_getLastErrorXml(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_lastErrorXml(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_getLastMethodSuccess(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putLastMethodSuccess(HCkAsnW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkAsnW_getNumSubItems(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void CkAsnW_getTag(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_tag(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC int CkAsnW_getTagValue(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_getVerboseLogging(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC void  CkAsnW_putVerboseLogging(HCkAsnW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkAsnW_getVersion(HCkAsnW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_version(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendBigInt(HCkAsnW cHandle, const wchar_t *encodedBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendBits(HCkAsnW cHandle, const wchar_t *encodedBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendBool(HCkAsnW cHandle, BOOL value);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendContextConstructed(HCkAsnW cHandle, int tag);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendContextPrimitive(HCkAsnW cHandle, int tag, const wchar_t *encodedBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendInt(HCkAsnW cHandle, int value);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendNull(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendOctets(HCkAsnW cHandle, const wchar_t *encodedBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendOid(HCkAsnW cHandle, const wchar_t *oid);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendSequence(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendSequence2(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC HCkAsnW CkAsnW_AppendSequenceR(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendSet(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendSet2(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC HCkAsnW CkAsnW_AppendSetR(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendString(HCkAsnW cHandle, const wchar_t *strType, const wchar_t *value);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AppendTime(HCkAsnW cHandle, const wchar_t *timeFormat, const wchar_t *dateTimeStr);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_AsnToXml(HCkAsnW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_asnToXml(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_DeleteSubItem(HCkAsnW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_GetBinaryDer(HCkAsnW cHandle, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_GetEncodedContent(HCkAsnW cHandle, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_getEncodedContent(HCkAsnW cHandle, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_GetEncodedDer(HCkAsnW cHandle, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkAsnW_getEncodedDer(HCkAsnW cHandle, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC HCkAsnW CkAsnW_GetLastSubItem(HCkAsnW cHandle);
CK_C_VISIBLE_PUBLIC HCkAsnW CkAsnW_GetSubItem(HCkAsnW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_LoadAsnXml(HCkAsnW cHandle, const wchar_t *xmlStr);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_LoadBd(HCkAsnW cHandle, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_LoadBinary(HCkAsnW cHandle, HCkByteData derBytes);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_LoadBinaryFile(HCkAsnW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_LoadEncoded(HCkAsnW cHandle, const wchar_t *asnContent, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_SaveLastError(HCkAsnW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_SetEncodedContent(HCkAsnW cHandle, const wchar_t *encodedBytes, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_WriteBd(HCkAsnW cHandle, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkAsnW_WriteBinaryDer(HCkAsnW cHandle, const wchar_t *path);
#endif
