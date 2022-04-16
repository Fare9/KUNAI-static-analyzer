// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPfxWH
#define _C_CkPfxWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPfxW CkPfxW_Create(void);
CK_C_VISIBLE_PUBLIC void CkPfxW_Dispose(HCkPfxW handle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getAlgorithmId(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putAlgorithmId(HCkPfxW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_algorithmId(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getDebugLogFilePath(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putDebugLogFilePath(HCkPfxW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_debugLogFilePath(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getLastErrorHtml(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_lastErrorHtml(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getLastErrorText(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_lastErrorText(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getLastErrorXml(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_lastErrorXml(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_getLastMethodSuccess(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putLastMethodSuccess(HCkPfxW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPfxW_getNumCerts(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC int CkPfxW_getNumPrivateKeys(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getPbes2CryptAlg(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putPbes2CryptAlg(HCkPfxW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_pbes2CryptAlg(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getPbes2HmacAlg(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putPbes2HmacAlg(HCkPfxW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_pbes2HmacAlg(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void CkPfxW_getUncommonOptions(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putUncommonOptions(HCkPfxW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_uncommonOptions(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_getVerboseLogging(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPfxW_putVerboseLogging(HCkPfxW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPfxW_getVersion(HCkPfxW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_version(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_AddCert(HCkPfxW cHandle, HCkCertW cert, BOOL includeChain);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_AddPrivateKey(HCkPfxW cHandle, HCkPrivateKeyW privKey, HCkCertChainW certChain);
CK_C_VISIBLE_PUBLIC HCkCertW CkPfxW_FindCertByLocalKeyId(HCkPfxW cHandle, const wchar_t *localKeyId, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC HCkCertW CkPfxW_GetCert(HCkPfxW cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkPrivateKeyW CkPfxW_GetPrivateKey(HCkPfxW cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_GetSafeBagAttr(HCkPfxW cHandle, BOOL forPrivateKey, int index, const wchar_t *attrName, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_getSafeBagAttr(HCkPfxW cHandle, BOOL forPrivateKey, int index, const wchar_t *attrName);
#if defined(CK_WINCERTSTORE_INCLUDED)
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ImportToWindows(HCkPfxW cHandle, BOOL exportable, BOOL userProtected, BOOL machineKeyset, BOOL allowOverwriteKey, BOOL allowExport, const wchar_t *leafStore, const wchar_t *intermediateStore, const wchar_t *rootStore, const wchar_t *extraOptions);
#endif
CK_C_VISIBLE_PUBLIC HCkJsonObjectW CkPfxW_LastJsonData(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_LoadPem(HCkPfxW cHandle, const wchar_t *pemStr, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_LoadPfxBytes(HCkPfxW cHandle, HCkByteData pfxData, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_LoadPfxEncoded(HCkPfxW cHandle, const wchar_t *encodedData, const wchar_t *encoding, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_LoadPfxFile(HCkPfxW cHandle, const wchar_t *path, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_SaveLastError(HCkPfxW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_SetSafeBagAttr(HCkPfxW cHandle, BOOL forPrivateKey, int index, const wchar_t *name, const wchar_t *value, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ToBinary(HCkPfxW cHandle, const wchar_t *password, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ToEncodedString(HCkPfxW cHandle, const wchar_t *password, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_toEncodedString(HCkPfxW cHandle, const wchar_t *password, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ToFile(HCkPfxW cHandle, const wchar_t *password, const wchar_t *path);
CK_C_VISIBLE_PUBLIC HCkJavaKeyStoreW CkPfxW_ToJavaKeyStore(HCkPfxW cHandle, const wchar_t *alias, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ToPem(HCkPfxW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_toPem(HCkPfxW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_ToPemEx(HCkPfxW cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const wchar_t *encryptAlg, const wchar_t *password, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPfxW_toPemEx(HCkPfxW cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const wchar_t *encryptAlg, const wchar_t *password);
CK_C_VISIBLE_PUBLIC BOOL CkPfxW_UseCertVault(HCkPfxW cHandle, HCkXmlCertVaultW vault);
#endif
