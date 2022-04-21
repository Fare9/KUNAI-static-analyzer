// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkEcc_H
#define _C_CkEcc_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkEcc CkEcc_Create(void);
CK_C_VISIBLE_PUBLIC void CkEcc_Dispose(HCkEcc handle);
CK_C_VISIBLE_PUBLIC void CkEcc_getDebugLogFilePath(HCkEcc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkEcc_putDebugLogFilePath(HCkEcc cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkEcc_debugLogFilePath(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_getLastErrorHtml(HCkEcc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEcc_lastErrorHtml(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_getLastErrorText(HCkEcc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEcc_lastErrorText(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_getLastErrorXml(HCkEcc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEcc_lastErrorXml(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_getLastMethodSuccess(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_putLastMethodSuccess(HCkEcc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_getUtf8(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_putUtf8(HCkEcc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_getVerboseLogging(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC void CkEcc_putVerboseLogging(HCkEcc cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkEcc_getVersion(HCkEcc cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEcc_version(HCkEcc cHandle);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkEcc_GenEccKey(HCkEcc cHandle, const char *curveName, HCkPrng prng);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkEcc_GenEccKey2(HCkEcc cHandle, const char *curveName, const char *encodedK, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_SaveLastError(HCkEcc cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_SharedSecretENC(HCkEcc cHandle, HCkPrivateKey privKey, HCkPublicKey pubKey, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkEcc_sharedSecretENC(HCkEcc cHandle, HCkPrivateKey privKey, HCkPublicKey pubKey, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_SignBd(HCkEcc cHandle, HCkBinData bdData, const char *hashAlg, const char *encoding, HCkPrivateKey privKey, HCkPrng prng, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkEcc_signBd(HCkEcc cHandle, HCkBinData bdData, const char *hashAlg, const char *encoding, HCkPrivateKey privKey, HCkPrng prng);
CK_C_VISIBLE_PUBLIC BOOL CkEcc_SignHashENC(HCkEcc cHandle, const char *encodedHash, const char *encoding, HCkPrivateKey privkey, HCkPrng prng, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkEcc_signHashENC(HCkEcc cHandle, const char *encodedHash, const char *encoding, HCkPrivateKey privkey, HCkPrng prng);
CK_C_VISIBLE_PUBLIC int CkEcc_VerifyBd(HCkEcc cHandle, HCkBinData bdData, const char *hashAlg, const char *encodedSig, const char *encoding, HCkPublicKey pubkey);
CK_C_VISIBLE_PUBLIC int CkEcc_VerifyHashENC(HCkEcc cHandle, const char *encodedHash, const char *encodedSig, const char *encoding, HCkPublicKey pubkey);
#endif
