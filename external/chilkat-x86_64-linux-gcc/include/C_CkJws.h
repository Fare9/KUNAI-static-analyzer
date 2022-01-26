// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJws_H
#define _C_CkJws_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJws CkJws_Create(void);
CK_C_VISIBLE_PUBLIC void CkJws_Dispose(HCkJws handle);
CK_C_VISIBLE_PUBLIC void CkJws_getDebugLogFilePath(HCkJws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkJws_putDebugLogFilePath(HCkJws cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkJws_debugLogFilePath(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_getLastErrorHtml(HCkJws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJws_lastErrorHtml(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_getLastErrorText(HCkJws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJws_lastErrorText(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_getLastErrorXml(HCkJws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJws_lastErrorXml(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJws_getLastMethodSuccess(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_putLastMethodSuccess(HCkJws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJws_getNumSignatures(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJws_getPreferCompact(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_putPreferCompact(HCkJws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJws_getPreferFlattened(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_putPreferFlattened(HCkJws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJws_getUtf8(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_putUtf8(HCkJws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJws_getVerboseLogging(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC void CkJws_putVerboseLogging(HCkJws cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJws_getVersion(HCkJws cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJws_version(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJws_CreateJws(HCkJws cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJws_createJws(HCkJws cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJws_CreateJwsSb(HCkJws cHandle, HCkStringBuilder sbJws);
CK_C_VISIBLE_PUBLIC BOOL CkJws_GetPayload(HCkJws cHandle, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJws_getPayload(HCkJws cHandle, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkJws_GetPayloadBd(HCkJws cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkJws_GetPayloadSb(HCkJws cHandle, const char *charset, HCkStringBuilder sbPayload);
CK_C_VISIBLE_PUBLIC HCkJsonObject CkJws_GetProtectedHeader(HCkJws cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkJsonObject CkJws_GetUnprotectedHeader(HCkJws cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJws_LoadJws(HCkJws cHandle, const char *jwsStr);
CK_C_VISIBLE_PUBLIC BOOL CkJws_LoadJwsSb(HCkJws cHandle, HCkStringBuilder sbJws);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SaveLastError(HCkJws cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetMacKey(HCkJws cHandle, int index, const char *key, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetMacKeyBd(HCkJws cHandle, int index, HCkBinData key);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetPayload(HCkJws cHandle, const char *payload, const char *charset, BOOL includeBom);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetPayloadBd(HCkJws cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetPayloadSb(HCkJws cHandle, HCkStringBuilder sbPayload, const char *charset, BOOL includeBom);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetPrivateKey(HCkJws cHandle, int index, HCkPrivateKey privKey);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetProtectedHeader(HCkJws cHandle, int index, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetPublicKey(HCkJws cHandle, int index, HCkPublicKey pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkJws_SetUnprotectedHeader(HCkJws cHandle, int index, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC int CkJws_Validate(HCkJws cHandle, int index);
#endif
