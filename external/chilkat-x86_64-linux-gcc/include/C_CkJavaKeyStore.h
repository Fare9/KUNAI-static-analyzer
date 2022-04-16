// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkJavaKeyStore_H
#define _C_CkJavaKeyStore_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkJavaKeyStore CkJavaKeyStore_Create(void);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_Dispose(HCkJavaKeyStore handle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_getDebugLogFilePath(HCkJavaKeyStore cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putDebugLogFilePath(HCkJavaKeyStore cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_debugLogFilePath(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_getLastErrorHtml(HCkJavaKeyStore cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_lastErrorHtml(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_getLastErrorText(HCkJavaKeyStore cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_lastErrorText(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_getLastErrorXml(HCkJavaKeyStore cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_lastErrorXml(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_getLastMethodSuccess(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putLastMethodSuccess(HCkJavaKeyStore cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStore_getNumPrivateKeys(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStore_getNumSecretKeys(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC int CkJavaKeyStore_getNumTrustedCerts(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_getRequireCompleteChain(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putRequireCompleteChain(HCkJavaKeyStore cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_getUtf8(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putUtf8(HCkJavaKeyStore cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_getVerboseLogging(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putVerboseLogging(HCkJavaKeyStore cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_getVerifyKeyedDigest(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_putVerifyKeyedDigest(HCkJavaKeyStore cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkJavaKeyStore_getVersion(HCkJavaKeyStore cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_version(HCkJavaKeyStore cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_AddPfx(HCkJavaKeyStore cHandle, HCkPfx pfx, const char *alias, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_AddPrivateKey(HCkJavaKeyStore cHandle, HCkCert cert, const char *alias, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_AddSecretKey(HCkJavaKeyStore cHandle, const char *encodedKeyBytes, const char *encoding, const char *algorithm, const char *alias, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_AddTrustedCert(HCkJavaKeyStore cHandle, HCkCert cert, const char *alias);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_ChangePassword(HCkJavaKeyStore cHandle, int index, const char *oldPassword, const char *newPassword);
CK_C_VISIBLE_PUBLIC HCkCertChain CkJavaKeyStore_FindCertChain(HCkJavaKeyStore cHandle, const char *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkJavaKeyStore_FindPrivateKey(HCkJavaKeyStore cHandle, const char *password, const char *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkCert CkJavaKeyStore_FindTrustedCert(HCkJavaKeyStore cHandle, const char *alias, BOOL caseSensitive);
CK_C_VISIBLE_PUBLIC HCkCertChain CkJavaKeyStore_GetCertChain(HCkJavaKeyStore cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkJavaKeyStore_GetPrivateKey(HCkJavaKeyStore cHandle, const char *password, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_GetPrivateKeyAlias(HCkJavaKeyStore cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_getPrivateKeyAlias(HCkJavaKeyStore cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_GetSecretKey(HCkJavaKeyStore cHandle, const char *password, int index, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_getSecretKey(HCkJavaKeyStore cHandle, const char *password, int index, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_GetSecretKeyAlias(HCkJavaKeyStore cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_getSecretKeyAlias(HCkJavaKeyStore cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkCert CkJavaKeyStore_GetTrustedCert(HCkJavaKeyStore cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_GetTrustedCertAlias(HCkJavaKeyStore cHandle, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_getTrustedCertAlias(HCkJavaKeyStore cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_LoadBd(HCkJavaKeyStore cHandle, const char *password, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_LoadBinary(HCkJavaKeyStore cHandle, const char *password, HCkByteData jksData);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_LoadEncoded(HCkJavaKeyStore cHandle, const char *password, const char *jksEncData, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_LoadFile(HCkJavaKeyStore cHandle, const char *password, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_LoadJwkSet(HCkJavaKeyStore cHandle, const char *password, HCkJsonObject jwkSet);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_RemoveEntry(HCkJavaKeyStore cHandle, int entryType, int index);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_SaveLastError(HCkJavaKeyStore cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_SetAlias(HCkJavaKeyStore cHandle, int entryType, int index, const char *alias);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_ToBinary(HCkJavaKeyStore cHandle, const char *password, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_ToEncodedString(HCkJavaKeyStore cHandle, const char *password, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkJavaKeyStore_toEncodedString(HCkJavaKeyStore cHandle, const char *password, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_ToFile(HCkJavaKeyStore cHandle, const char *password, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_ToJwkSet(HCkJavaKeyStore cHandle, const char *password, HCkStringBuilder sbJwkSet);
CK_C_VISIBLE_PUBLIC HCkPem CkJavaKeyStore_ToPem(HCkJavaKeyStore cHandle, const char *password);
CK_C_VISIBLE_PUBLIC HCkPfx CkJavaKeyStore_ToPfx(HCkJavaKeyStore cHandle, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_UnlockComponent(HCkJavaKeyStore cHandle, const char *unlockCode);
CK_C_VISIBLE_PUBLIC BOOL CkJavaKeyStore_UseCertVault(HCkJavaKeyStore cHandle, HCkXmlCertVault vault);
#endif
