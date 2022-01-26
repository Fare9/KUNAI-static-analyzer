// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkScMinidriver_H
#define _C_CkScMinidriver_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkScMinidriver CkScMinidriver_Create(void);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_Dispose(HCkScMinidriver handle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getAtr(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_atr(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getCardName(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_cardName(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getDebugLogFilePath(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putDebugLogFilePath(HCkScMinidriver cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_debugLogFilePath(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getLastErrorHtml(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_lastErrorHtml(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getLastErrorText(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_lastErrorText(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getLastErrorXml(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_lastErrorXml(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_getLastMethodSuccess(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putLastMethodSuccess(HCkScMinidriver cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkScMinidriver_getMaxContainers(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getRsaPaddingHash(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putRsaPaddingHash(HCkScMinidriver cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_rsaPaddingHash(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getRsaPaddingScheme(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putRsaPaddingScheme(HCkScMinidriver cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_rsaPaddingScheme(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getUncommonOptions(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putUncommonOptions(HCkScMinidriver cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_uncommonOptions(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_getUtf8(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putUtf8(HCkScMinidriver cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_getVerboseLogging(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_putVerboseLogging(HCkScMinidriver cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScMinidriver_getVersion(HCkScMinidriver cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkScMinidriver_version(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_AcquireContext(HCkScMinidriver cHandle, const char *readerName);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_CardDeleteFile(HCkScMinidriver cHandle, const char *dirName, const char *fileName);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_DeleteCert(HCkScMinidriver cHandle, HCkCert cert, BOOL delPrivKey);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_DeleteContext(HCkScMinidriver cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_DeleteKeyContainer(HCkScMinidriver cHandle, int containerIndex);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_EnumFiles(HCkScMinidriver cHandle, const char *dirName, HCkStringTable st);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_FindCert(HCkScMinidriver cHandle, const char *certPart, const char *partValue, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_GenerateKey(HCkScMinidriver cHandle, int containerIndex, const char *keySpec, const char *keyType, int keySize, const char *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_GetCardProperties(HCkScMinidriver cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_GetCert(HCkScMinidriver cHandle, int containerIndex, const char *keySpec, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_GetContainerKeys(HCkScMinidriver cHandle, int containerIndex, HCkPublicKey sigKey, HCkPublicKey kexKey);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_GetCspContainerMap(HCkScMinidriver cHandle, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_ImportCert(HCkScMinidriver cHandle, HCkCert cert, int containerIndex, const char *keySpec, const char *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_ImportKey(HCkScMinidriver cHandle, int containerIndex, const char *keySpec, HCkPrivateKey privKey, const char *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_ListCerts(HCkScMinidriver cHandle, const char *certPart, HCkStringTable st);
CK_C_VISIBLE_PUBLIC int CkScMinidriver_PinAuthenticate(HCkScMinidriver cHandle, const char *pinId, const char *pin);
CK_C_VISIBLE_PUBLIC int CkScMinidriver_PinAuthenticateHex(HCkScMinidriver cHandle, const char *pinId, const char *pin);
CK_C_VISIBLE_PUBLIC int CkScMinidriver_PinChange(HCkScMinidriver cHandle, const char *pinId, const char *currentPin, const char *newPin);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_PinDeauthenticate(HCkScMinidriver cHandle, const char *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_ReadFile(HCkScMinidriver cHandle, const char *dirName, const char *fileName, HCkBinData bd);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_SaveLastError(HCkScMinidriver cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_SignData(HCkScMinidriver cHandle, int containerIndex, const char *keySpec, const char *hashDataAlg, HCkBinData bdData, HCkBinData bdSignedData);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriver_WriteFile(HCkScMinidriver cHandle, const char *dirName, const char *fileName, HCkBinData bd);
#endif
