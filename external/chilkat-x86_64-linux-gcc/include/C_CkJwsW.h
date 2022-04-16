// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJwsWH
#define _C_CkJwsWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJwsW CkJwsW_Create(void);
CK_C_VISIBLE_PUBLIC void CkJwsW_Dispose(HCkJwsW handle);
CK_C_VISIBLE_PUBLIC void CkJwsW_getDebugLogFilePath(HCkJwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkJwsW_putDebugLogFilePath(HCkJwsW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_debugLogFilePath(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkJwsW_getLastErrorHtml(HCkJwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_lastErrorHtml(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkJwsW_getLastErrorText(HCkJwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_lastErrorText(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void CkJwsW_getLastErrorXml(HCkJwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_lastErrorXml(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_getLastMethodSuccess(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJwsW_putLastMethodSuccess(HCkJwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJwsW_getNumSignatures(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_getPreferCompact(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJwsW_putPreferCompact(HCkJwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_getPreferFlattened(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJwsW_putPreferFlattened(HCkJwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_getVerboseLogging(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJwsW_putVerboseLogging(HCkJwsW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJwsW_getVersion(HCkJwsW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_version(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_CreateJws(HCkJwsW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_createJws(HCkJwsW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_CreateJwsSb(HCkJwsW cHandle, HCkStringBuilderW sbJws);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_GetPayload(HCkJwsW cHandle, const wchar_t *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJwsW_getPayload(HCkJwsW cHandle, const wchar_t *charset);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_GetPayloadBd(HCkJwsW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_GetPayloadSb(HCkJwsW cHandle, const wchar_t *charset, HCkStringBuilderW sbPayload);
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkJwsW_GetProtectedHeader(HCkJwsW cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkJwsW_GetUnprotectedHeader(HCkJwsW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_LoadJws(HCkJwsW cHandle, const wchar_t *jwsStr);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_LoadJwsSb(HCkJwsW cHandle, HCkStringBuilderW sbJws);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SaveLastError(HCkJwsW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetMacKey(HCkJwsW cHandle, int index, const wchar_t *key, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetMacKeyBd(HCkJwsW cHandle, int index, HCkBinDataW key);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetPayload(HCkJwsW cHandle, const wchar_t *payload, const wchar_t *charset, BOOL includeBom);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetPayloadBd(HCkJwsW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetPayloadSb(HCkJwsW cHandle, HCkStringBuilderW sbPayload, const wchar_t *charset, BOOL includeBom);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetPrivateKey(HCkJwsW cHandle, int index, HCkPrivateKeyW privKey);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetProtectedHeader(HCkJwsW cHandle, int index, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetPublicKey(HCkJwsW cHandle, int index, HCkPublicKeyW pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkJwsW_SetUnprotectedHeader(HCkJwsW cHandle, int index, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC int CkJwsW_Validate(HCkJwsW cHandle, int index);
#endif
