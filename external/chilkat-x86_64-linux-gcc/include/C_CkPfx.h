// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPfx_H
#define _C_CkPfx_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPfx CkPfx_Create(void);
CK_C_VISIBLE_PUBLIC void CkPfx_Dispose(HCkPfx handle);
CK_C_VISIBLE_PUBLIC void CkPfx_getAlgorithmId(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPfx_putAlgorithmId(HCkPfx cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPfx_algorithmId(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getDebugLogFilePath(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPfx_putDebugLogFilePath(HCkPfx cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPfx_debugLogFilePath(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getLastErrorHtml(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPfx_lastErrorHtml(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getLastErrorText(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPfx_lastErrorText(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getLastErrorXml(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPfx_lastErrorXml(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_getLastMethodSuccess(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_putLastMethodSuccess(HCkPfx cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPfx_getNumCerts(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC int CkPfx_getNumPrivateKeys(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getPbes2CryptAlg(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPfx_putPbes2CryptAlg(HCkPfx cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPfx_pbes2CryptAlg(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getPbes2HmacAlg(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPfx_putPbes2HmacAlg(HCkPfx cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPfx_pbes2HmacAlg(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_getUncommonOptions(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPfx_putUncommonOptions(HCkPfx cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPfx_uncommonOptions(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_getUtf8(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_putUtf8(HCkPfx cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_getVerboseLogging(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC void CkPfx_putVerboseLogging(HCkPfx cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPfx_getVersion(HCkPfx cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPfx_version(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_AddCert(HCkPfx cHandle, HCkCert cert, BOOL includeChain);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_AddPrivateKey(HCkPfx cHandle, HCkPrivateKey privKey, HCkCertChain certChain);
CK_C_VISIBLE_PUBLIC HCkCert CkPfx_FindCertByLocalKeyId(HCkPfx cHandle, const char *localKeyId, const char *encoding);
CK_C_VISIBLE_PUBLIC HCkCert CkPfx_GetCert(HCkPfx cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkPfx_GetPrivateKey(HCkPfx cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_GetSafeBagAttr(HCkPfx cHandle, BOOL forPrivateKey, int index, const char *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPfx_getSafeBagAttr(HCkPfx cHandle, BOOL forPrivateKey, int index, const char *attrName);
#if defined(CK_WINCERTSTORE_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ImportToWindows(HCkPfx cHandle, BOOL exportable, BOOL userProtected, BOOL machineKeyset, BOOL allowOverwriteKey, BOOL allowExport, const char *leafStore, const char *intermediateStore, const char *rootStore, const char *extraOptions);
#endif
CK_C_VISIBLE_PUBLIC HCkJsonObject CkPfx_LastJsonData(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_LoadPem(HCkPfx cHandle, const char *pemStr, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_LoadPfxBytes(HCkPfx cHandle, HCkByteData pfxData, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_LoadPfxEncoded(HCkPfx cHandle, const char *encodedData, const char *encoding, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_LoadPfxFile(HCkPfx cHandle, const char *path, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_SaveLastError(HCkPfx cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_SetSafeBagAttr(HCkPfx cHandle, BOOL forPrivateKey, int index, const char *name, const char *value, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ToBinary(HCkPfx cHandle, const char *password, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ToEncodedString(HCkPfx cHandle, const char *password, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPfx_toEncodedString(HCkPfx cHandle, const char *password, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ToFile(HCkPfx cHandle, const char *password, const char *path);
CK_C_VISIBLE_PUBLIC HCkJavaKeyStore CkPfx_ToJavaKeyStore(HCkPfx cHandle, const char *alias, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ToPem(HCkPfx cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPfx_toPem(HCkPfx cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_ToPemEx(HCkPfx cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const char *encryptAlg, const char *password, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPfx_toPemEx(HCkPfx cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const char *encryptAlg, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfx_UseCertVault(HCkPfx cHandle, HCkXmlCertVault vault);
#endif
