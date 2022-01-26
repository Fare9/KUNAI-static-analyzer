// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkEdDSAWH
#define _C_CkEdDSAWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkEdDSAW CkEdDSAW_Create(void);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_Dispose(HCkEdDSAW handle);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_getDebugLogFilePath(HCkEdDSAW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkEdDSAW_putDebugLogFilePath(HCkEdDSAW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_debugLogFilePath(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_getLastErrorHtml(HCkEdDSAW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_lastErrorHtml(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_getLastErrorText(HCkEdDSAW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_lastErrorText(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_getLastErrorXml(HCkEdDSAW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_lastErrorXml(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_getLastMethodSuccess(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC void  CkEdDSAW_putLastMethodSuccess(HCkEdDSAW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_getVerboseLogging(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC void  CkEdDSAW_putVerboseLogging(HCkEdDSAW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkEdDSAW_getVersion(HCkEdDSAW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_version(HCkEdDSAW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_GenEd25519Key(HCkEdDSAW cHandle, HCkPrngW prng, HCkPrivateKeyW privKey);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_SaveLastError(HCkEdDSAW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_SharedSecretENC(HCkEdDSAW cHandle, HCkPrivateKeyW privkey, HCkPublicKeyW pubkey, const wchar_t *encoding, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_sharedSecretENC(HCkEdDSAW cHandle, HCkPrivateKeyW privkey, HCkPublicKeyW pubkey, const wchar_t *encoding);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_SignBdENC(HCkEdDSAW cHandle, HCkBinDataW bd, const wchar_t *encoding, HCkPrivateKeyW privkey, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkEdDSAW_signBdENC(HCkEdDSAW cHandle, HCkBinDataW bd, const wchar_t *encoding, HCkPrivateKeyW privkey);
CK_C_VISIBLE_PUBLIC BOOL CkEdDSAW_VerifyBdENC(HCkEdDSAW cHandle, HCkBinDataW bd, const wchar_t *encodedSig, const wchar_t *enocding, HCkPublicKeyW pubkey);
#endif
