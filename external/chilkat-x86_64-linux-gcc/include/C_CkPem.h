// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPem_H
#define _C_CkPem_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkPem_setAbortCheck(HCkPem cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkPem_setPercentDone(HCkPem cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkPem_setProgressInfo(HCkPem cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkPem_setTaskCompleted(HCkPem cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkPem_setAbortCheck2(HCkPem cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkPem_setPercentDone2(HCkPem cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkPem_setProgressInfo2(HCkPem cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkPem_setTaskCompleted2(HCkPem cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkPem_setExternalProgress(HCkPem cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkPem_setCallbackContext(HCkPem cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkPem CkPem_Create(void);
CK_C_VISIBLE_PUBLIC void CkPem_Dispose(HCkPem handle);
CK_C_VISIBLE_PUBLIC BOOL CkPem_getAppendMode(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_putAppendMode(HCkPem cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPem_getDebugLogFilePath(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPem_putDebugLogFilePath(HCkPem cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPem_debugLogFilePath(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC int CkPem_getHeartbeatMs(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_putHeartbeatMs(HCkPem cHandle, int newVal);
CK_C_VISIBLE_PUBLIC void CkPem_getLastErrorHtml(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPem_lastErrorHtml(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_getLastErrorText(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPem_lastErrorText(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_getLastErrorXml(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPem_lastErrorXml(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPem_getLastMethodSuccess(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_putLastMethodSuccess(HCkPem cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPem_getNumCerts(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC int CkPem_getNumCrls(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC int CkPem_getNumCsrs(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC int CkPem_getNumPrivateKeys(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC int CkPem_getNumPublicKeys(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_getPrivateKeyFormat(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPem_putPrivateKeyFormat(HCkPem cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPem_privateKeyFormat(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_getPublicKeyFormat(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPem_putPublicKeyFormat(HCkPem cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPem_publicKeyFormat(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPem_getUtf8(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_putUtf8(HCkPem cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPem_getVerboseLogging(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC void CkPem_putVerboseLogging(HCkPem cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPem_getVersion(HCkPem cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPem_version(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPem_AddCert(HCkPem cHandle, HCkCert cert, BOOL includeChain);
CK_C_VISIBLE_PUBLIC BOOL CkPem_AddItem(HCkPem cHandle, const char *itemType, const char *encoding, const char *itemData);
CK_C_VISIBLE_PUBLIC BOOL CkPem_AddPrivateKey(HCkPem cHandle, HCkPrivateKey privateKey);
CK_C_VISIBLE_PUBLIC BOOL CkPem_AddPrivateKey2(HCkPem cHandle, HCkPrivateKey privKey, HCkCertChain certChain);
CK_C_VISIBLE_PUBLIC BOOL CkPem_AddPublicKey(HCkPem cHandle, HCkPublicKey pubkey);
CK_C_VISIBLE_PUBLIC BOOL CkPem_Clear(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC HCkCert CkPem_GetCert(HCkPem cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPem_GetEncodedItem(HCkPem cHandle, const char *itemType, const char *itemSubType, const char *encoding, int index, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPem_getEncodedItem(HCkPem cHandle, const char *itemType, const char *itemSubType, const char *encoding, int index);
CK_C_VISIBLE_PUBLIC HCkPrivateKey CkPem_GetPrivateKey(HCkPem cHandle, int index);
CK_C_VISIBLE_PUBLIC HCkPublicKey CkPem_GetPublicKey(HCkPem cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPem_LoadP7b(HCkPem cHandle, HCkByteData p7bData);
CK_C_VISIBLE_PUBLIC HCkTask CkPem_LoadP7bAsync(HCkPem cHandle, HCkByteData p7bData);
CK_C_VISIBLE_PUBLIC BOOL CkPem_LoadP7bFile(HCkPem cHandle, const char *path);
CK_C_VISIBLE_PUBLIC HCkTask CkPem_LoadP7bFileAsync(HCkPem cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkPem_LoadPem(HCkPem cHandle, const char *pemContent, const char *password);
CK_C_VISIBLE_PUBLIC HCkTask CkPem_LoadPemAsync(HCkPem cHandle, const char *pemContent, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPem_LoadPemFile(HCkPem cHandle, const char *path, const char *password);
CK_C_VISIBLE_PUBLIC HCkTask CkPem_LoadPemFileAsync(HCkPem cHandle, const char *path, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPem_LoadTaskCaller(HCkPem cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkPem_RemoveCert(HCkPem cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPem_RemovePrivateKey(HCkPem cHandle, int index);
CK_C_VISIBLE_PUBLIC BOOL CkPem_SaveLastError(HCkPem cHandle, const char *path);
CK_C_VISIBLE_PUBLIC HCkJavaKeyStore CkPem_ToJks(HCkPem cHandle, const char *alias, const char *password);
CK_C_VISIBLE_PUBLIC BOOL CkPem_ToPem(HCkPem cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPem_toPem(HCkPem cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPem_ToPemEx(HCkPem cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const char *encryptAlg, const char *password, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkPem_toPemEx(HCkPem cHandle, BOOL extendedAttrs, BOOL noKeys, BOOL noCerts, BOOL noCaCerts, const char *encryptAlg, const char *password);
CK_C_VISIBLE_PUBLIC HCkPfx CkPem_ToPfx(HCkPem cHandle);
#endif
