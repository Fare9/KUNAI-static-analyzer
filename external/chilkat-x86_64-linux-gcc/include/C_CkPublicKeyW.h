// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPublicKeyWH
#define _C_CkPublicKeyWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPublicKeyW CkPublicKeyW_Create(void);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_Dispose(HCkPublicKeyW handle);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getDebugLogFilePath(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkPublicKeyW_putDebugLogFilePath(HCkPublicKeyW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_debugLogFilePath(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_getEmpty(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC int CkPublicKeyW_getKeySize(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getKeyType(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_keyType(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getLastErrorHtml(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_lastErrorHtml(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getLastErrorText(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_lastErrorText(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getLastErrorXml(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_lastErrorXml(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_getLastMethodSuccess(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPublicKeyW_putLastMethodSuccess(HCkPublicKeyW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_getVerboseLogging(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC void  CkPublicKeyW_putVerboseLogging(HCkPublicKeyW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPublicKeyW_getVersion(HCkPublicKeyW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_version(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetDer(HCkPublicKeyW cHandle, BOOL preferPkcs1, HCkByteData outBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetEncoded(HCkPublicKeyW cHandle, BOOL preferPkcs1, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getEncoded(HCkPublicKeyW cHandle, BOOL preferPkcs1, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetJwk(HCkPublicKeyW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getJwk(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetJwkThumbprint(HCkPublicKeyW cHandle, const wchar_t *hashAlg, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getJwkThumbprint(HCkPublicKeyW cHandle, const wchar_t *hashAlg);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetOpenSslDer(HCkPublicKeyW cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetOpenSslPem(HCkPublicKeyW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getOpenSslPem(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetPem(HCkPublicKeyW cHandle, BOOL preferPkcs1, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getPem(HCkPublicKeyW cHandle, BOOL preferPkcs1);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetPkcs1ENC(HCkPublicKeyW cHandle, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getPkcs1ENC(HCkPublicKeyW cHandle, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetPkcs8ENC(HCkPublicKeyW cHandle, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getPkcs8ENC(HCkPublicKeyW cHandle, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetRsaDer(HCkPublicKeyW cHandle, HCkByteData outData);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_GetXml(HCkPublicKeyW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkPublicKeyW_getXml(HCkPublicKeyW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadBase64(HCkPublicKeyW cHandle, const wchar_t *keyStr);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadBd(HCkPublicKeyW cHandle, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadEcdsa(HCkPublicKeyW cHandle, const wchar_t *curveName, const wchar_t *Qx, const wchar_t *Qy);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadEd25519(HCkPublicKeyW cHandle, const wchar_t *pubKey);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadFromBinary(HCkPublicKeyW cHandle, HCkByteData keyBytes);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadFromFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadFromString(HCkPublicKeyW cHandle, const wchar_t *keyString);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadOpenSslDer(HCkPublicKeyW cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadOpenSslDerFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadOpenSslPem(HCkPublicKeyW cHandle, const wchar_t *str);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadOpenSslPemFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadPkcs1Pem(HCkPublicKeyW cHandle, const wchar_t *str);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadRsaDer(HCkPublicKeyW cHandle, HCkByteData data);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadRsaDerFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadXml(HCkPublicKeyW cHandle, const wchar_t *xml);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_LoadXmlFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveDerFile(HCkPublicKeyW cHandle, BOOL preferPkcs1, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveLastError(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveOpenSslDerFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveOpenSslPemFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SavePemFile(HCkPublicKeyW cHandle, BOOL preferPkcs1, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveRsaDerFile(HCkPublicKeyW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkPublicKeyW_SaveXmlFile(HCkPublicKeyW cHandle, const wchar_t *path);
#endif
