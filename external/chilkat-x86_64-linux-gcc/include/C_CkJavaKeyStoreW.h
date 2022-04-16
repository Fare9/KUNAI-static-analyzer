// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJavaKeyStoreWH
#define _C_CkJavaKeyStoreWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJavaKeyStoreW CkJavaKeyStoreW_Create(void);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_Dispose(HCkJavaKeyStoreW handle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_getDebugLogFilePath(HCkJavaKeyStoreW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkJavaKeyStoreW_putDebugLogFilePath(HCkJavaKeyStoreW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_debugLogFilePath(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_getLastErrorHtml(HCkJavaKeyStoreW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_lastErrorHtml(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_getLastErrorText(HCkJavaKeyStoreW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_lastErrorText(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_getLastErrorXml(HCkJavaKeyStoreW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_lastErrorXml(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_getLastMethodSuccess(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJavaKeyStoreW_putLastMethodSuccess(HCkJavaKeyStoreW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStoreW_getNumPrivateKeys(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStoreW_getNumSecretKeys(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStoreW_getNumTrustedCerts(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_getRequireCompleteChain(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJavaKeyStoreW_putRequireCompleteChain(HCkJavaKeyStoreW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_getVerboseLogging(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJavaKeyStoreW_putVerboseLogging(HCkJavaKeyStoreW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_getVerifyKeyedDigest(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC void  CkJavaKeyStoreW_putVerifyKeyedDigest(HCkJavaKeyStoreW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStoreW_getVersion(HCkJavaKeyStoreW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_version(HCkJavaKeyStoreW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_AddPfx(HCkJavaKeyStoreW cHandle, HCkPfxW pfx, const wchar_t *alias, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_AddPrivateKey(HCkJavaKeyStoreW cHandle, HCkCertW cert, const wchar_t *alias, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_AddSecretKey(HCkJavaKeyStoreW cHandle, const wchar_t *encodedKeyBytes, const wchar_t *encoding, const wchar_t *algorithm, const wchar_t *alias, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_AddTrustedCert(HCkJavaKeyStoreW cHandle, HCkCertW cert, const wchar_t *alias);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_ChangePassword(HCkJavaKeyStoreW cHandle, int index, const wchar_t *oldPassword, const wchar_t *newPassword);
CK_C_VISIBLE_PUBLIC HCkCertChainW CkJavaKeyStoreW_FindCertChain(HCkJavaKeyStoreW cHandle, const wchar_t *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkPrivateKeyW CkJavaKeyStoreW_FindPrivateKey(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkCertW CkJavaKeyStoreW_FindTrustedCert(HCkJavaKeyStoreW cHandle, const wchar_t *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkCertChainW CkJavaKeyStoreW_GetCertChain(HCkJavaKeyStoreW cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkPrivateKeyW CkJavaKeyStoreW_GetPrivateKey(HCkJavaKeyStoreW cHandle, const wchar_t *password, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_GetPrivateKeyAlias(HCkJavaKeyStoreW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_getPrivateKeyAlias(HCkJavaKeyStoreW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_GetSecretKey(HCkJavaKeyStoreW cHandle, const wchar_t *password, int index, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_getSecretKey(HCkJavaKeyStoreW cHandle, const wchar_t *password, int index, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_GetSecretKeyAlias(HCkJavaKeyStoreW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_getSecretKeyAlias(HCkJavaKeyStoreW cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkCertW CkJavaKeyStoreW_GetTrustedCert(HCkJavaKeyStoreW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_GetTrustedCertAlias(HCkJavaKeyStoreW cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_getTrustedCertAlias(HCkJavaKeyStoreW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_LoadBd(HCkJavaKeyStoreW cHandle, const wchar_t *password, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_LoadBinary(HCkJavaKeyStoreW cHandle, const wchar_t *password, HCkByteData jksData);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_LoadEncoded(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *jksEncData, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_LoadFile(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_LoadJwkSet(HCkJavaKeyStoreW cHandle, const wchar_t *password, HCkJsonObjectW jwkSet);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_RemoveEntry(HCkJavaKeyStoreW cHandle, int entryType, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_SaveLastError(HCkJavaKeyStoreW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_SetAlias(HCkJavaKeyStoreW cHandle, int entryType, int index, const wchar_t *alias);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_ToBinary(HCkJavaKeyStoreW cHandle, const wchar_t *password, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_ToEncodedString(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkJavaKeyStoreW_toEncodedString(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_ToFile(HCkJavaKeyStoreW cHandle, const wchar_t *password, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_ToJwkSet(HCkJavaKeyStoreW cHandle, const wchar_t *password, HCkStringBuilderW sbJwkSet);
CK_C_VISIBLE_PUBLIC HCkPemW CkJavaKeyStoreW_ToPem(HCkJavaKeyStoreW cHandle, const wchar_t *password);
CK_C_VISIBLE_PUBLIC HCkPfxW CkJavaKeyStoreW_ToPfx(HCkJavaKeyStoreW cHandle, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_UnlockComponent(HCkJavaKeyStoreW cHandle, const wchar_t *unlockCode);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStoreW_UseCertVault(HCkJavaKeyStoreW cHandle, HCkXmlCertVaultW vault);
#endif
