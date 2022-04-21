// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJwe_H
#define _C_CkJwe_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJwe CkJwe_Create(void);
CK_C_VISIBLE_PUBLIC void CkJwe_Dispose(HCkJwe handle);
CK_C_VISIBLE_PUBLIC void CkJwe_getDebugLogFilePath(HCkJwe cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkJwe_putDebugLogFilePath(HCkJwe cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkJwe_debugLogFilePath(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_getLastErrorHtml(HCkJwe cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwe_lastErrorHtml(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_getLastErrorText(HCkJwe cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwe_lastErrorText(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_getLastErrorXml(HCkJwe cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwe_lastErrorXml(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_getLastMethodSuccess(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_putLastMethodSuccess(HCkJwe cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJwe_getNumRecipients(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_getPreferCompact(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_putPreferCompact(HCkJwe cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_getPreferFlattened(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_putPreferFlattened(HCkJwe cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_getUtf8(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_putUtf8(HCkJwe cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_getVerboseLogging(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC void CkJwe_putVerboseLogging(HCkJwe cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJwe_getVersion(HCkJwe cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJwe_version(HCkJwe cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_Decrypt(HCkJwe cHandle, int index, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwe_decrypt(HCkJwe cHandle, int index, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_DecryptBd(HCkJwe cHandle, int index, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_DecryptSb(HCkJwe cHandle, int index, const char *charset, HCkStringBuilder contentSb);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_Encrypt(HCkJwe cHandle, const char *content, const char *charset, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJwe_encrypt(HCkJwe cHandle, const char *content, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_EncryptBd(HCkJwe cHandle, HCkBinData contentBd, HCkStringBuilder jweSb);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_EncryptSb(HCkJwe cHandle, HCkStringBuilder contentSb, const char *charset, HCkStringBuilder jweSb);
CK_C_VISIBLE_PUBLIC int CkJwe_FindRecipient(HCkJwe cHandle, const char *paramName, const char *paramValue, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_GetHeader(HCkJwe cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_GetProtectedHeader(HCkJwe cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_LoadJwe(HCkJwe cHandle, const char *jwe);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_LoadJweSb(HCkJwe cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SaveLastError(HCkJwe cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetAad(HCkJwe cHandle, const char *aad, const char *charset);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetAadBd(HCkJwe cHandle, HCkBinData aad);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetPassword(HCkJwe cHandle, int index, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetPrivateKey(HCkJwe cHandle, int index, HCkPrivateKey privKey);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetProtectedHeader(HCkJwe cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetPublicKey(HCkJwe cHandle, int index, HCkPublicKey pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetRecipientHeader(HCkJwe cHandle, int index, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetUnprotectedHeader(HCkJwe cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkJwe_SetWrappingKey(HCkJwe cHandle, int index, const char *encodedKey, const char *encoding);
#endif
