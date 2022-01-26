// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkScMinidriverWH
#define _C_CkScMinidriverWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkScMinidriverW CkScMinidriverW_Create(void);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_Dispose(HCkScMinidriverW handle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getAtr(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_atr(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getCardName(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_cardName(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getDebugLogFilePath(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putDebugLogFilePath(HCkScMinidriverW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_debugLogFilePath(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getLastErrorHtml(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_lastErrorHtml(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getLastErrorText(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_lastErrorText(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getLastErrorXml(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_lastErrorXml(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_getLastMethodSuccess(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putLastMethodSuccess(HCkScMinidriverW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkScMinidriverW_getMaxContainers(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getRsaPaddingHash(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putRsaPaddingHash(HCkScMinidriverW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_rsaPaddingHash(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getRsaPaddingScheme(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putRsaPaddingScheme(HCkScMinidriverW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_rsaPaddingScheme(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getUncommonOptions(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putUncommonOptions(HCkScMinidriverW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_uncommonOptions(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_getVerboseLogging(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC void  CkScMinidriverW_putVerboseLogging(HCkScMinidriverW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkScMinidriverW_getVersion(HCkScMinidriverW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkScMinidriverW_version(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_AcquireContext(HCkScMinidriverW cHandle, const wchar_t *readerName);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_CardDeleteFile(HCkScMinidriverW cHandle, const wchar_t *dirName, const wchar_t *fileName);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_DeleteCert(HCkScMinidriverW cHandle, HCkCertW cert, BOOL delPrivKey);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_DeleteContext(HCkScMinidriverW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_DeleteKeyContainer(HCkScMinidriverW cHandle, int containerIndex);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_EnumFiles(HCkScMinidriverW cHandle, const wchar_t *dirName, HCkStringTableW st);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_FindCert(HCkScMinidriverW cHandle, const wchar_t *certPart, const wchar_t *partValue, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_GenerateKey(HCkScMinidriverW cHandle, int containerIndex, const wchar_t *keySpec, const wchar_t *keyType, int keySize, const wchar_t *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_GetCardProperties(HCkScMinidriverW cHandle, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_GetCert(HCkScMinidriverW cHandle, int containerIndex, const wchar_t *keySpec, HCkCertW cert);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_GetContainerKeys(HCkScMinidriverW cHandle, int containerIndex, HCkPublicKeyW sigKey, HCkPublicKeyW kexKey);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_GetCspContainerMap(HCkScMinidriverW cHandle, HCkJsonObjectW json);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_ImportCert(HCkScMinidriverW cHandle, HCkCertW cert, int containerIndex, const wchar_t *keySpec, const wchar_t *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_ImportKey(HCkScMinidriverW cHandle, int containerIndex, const wchar_t *keySpec, HCkPrivateKeyW privKey, const wchar_t *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_ListCerts(HCkScMinidriverW cHandle, const wchar_t *certPart, HCkStringTableW st);
CK_C_VISIBLE_PUBLIC int CkScMinidriverW_PinAuthenticate(HCkScMinidriverW cHandle, const wchar_t *pinId, const wchar_t *pin);
CK_C_VISIBLE_PUBLIC int CkScMinidriverW_PinAuthenticateHex(HCkScMinidriverW cHandle, const wchar_t *pinId, const wchar_t *pin);
CK_C_VISIBLE_PUBLIC int CkScMinidriverW_PinChange(HCkScMinidriverW cHandle, const wchar_t *pinId, const wchar_t *currentPin, const wchar_t *newPin);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_PinDeauthenticate(HCkScMinidriverW cHandle, const wchar_t *pinId);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_ReadFile(HCkScMinidriverW cHandle, const wchar_t *dirName, const wchar_t *fileName, HCkBinDataW bd);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_SaveLastError(HCkScMinidriverW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_SignData(HCkScMinidriverW cHandle, int containerIndex, const wchar_t *keySpec, const wchar_t *hashDataAlg, HCkBinDataW bdData, HCkBinDataW bdSignedData);
CK_C_VISIBLE_PUBLIC BOOL CkScMinidriverW_WriteFile(HCkScMinidriverW cHandle, const wchar_t *dirName, const wchar_t *fileName, HCkBinDataW bd);
#endif
