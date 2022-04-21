// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkEdDSA_H
#define _C_CkEdDSA_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkEdDSA CkEdDSA_Create(void);
CK_C_VISIBLE_PUBLIC void CkEdDSA_Dispose(HCkEdDSA handle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_getDebugLogFilePath(HCkEdDSA cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkEdDSA_putDebugLogFilePath(HCkEdDSA cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_debugLogFilePath(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_getLastErrorHtml(HCkEdDSA cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_lastErrorHtml(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_getLastErrorText(HCkEdDSA cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_lastErrorText(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_getLastErrorXml(HCkEdDSA cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_lastErrorXml(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_getLastMethodSuccess(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_putLastMethodSuccess(HCkEdDSA cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_getUtf8(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_putUtf8(HCkEdDSA cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_getVerboseLogging(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSA_putVerboseLogging(HCkEdDSA cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkEdDSA_getVersion(HCkEdDSA cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_version(HCkEdDSA cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_GenEd25519Key(HCkEdDSA cHandle, HCkPrng prng, HCkPrivateKey privKey);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_SaveLastError(HCkEdDSA cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_SharedSecretENC(HCkEdDSA cHandle, HCkPrivateKey privkey, HCkPublicKey pubkey, const char *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_sharedSecretENC(HCkEdDSA cHandle, HCkPrivateKey privkey, HCkPublicKey pubkey, const char *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_SignBdENC(HCkEdDSA cHandle, HCkBinData bd, const char *encoding, HCkPrivateKey privkey, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkEdDSA_signBdENC(HCkEdDSA cHandle, HCkBinData bd, const char *encoding, HCkPrivateKey privkey);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSA_VerifyBdENC(HCkEdDSA cHandle, HCkBinData bd, const char *encodedSig, const char *enocding, HCkPublicKey pubkey);
#endif
