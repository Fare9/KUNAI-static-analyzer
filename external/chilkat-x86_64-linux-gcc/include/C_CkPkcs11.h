// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkPkcs11_H
#define _C_CkPkcs11_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkPkcs11 CkPkcs11_Create(void);
CK_C_VISIBLE_PUBLIC void CkPkcs11_Dispose(HCkPkcs11 handle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getDebugLogFilePath(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPkcs11_putDebugLogFilePath(HCkPkcs11 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_debugLogFilePath(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getLastErrorHtml(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_lastErrorHtml(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getLastErrorText(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_lastErrorText(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getLastErrorXml(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_lastErrorXml(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_getLastMethodSuccess(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_putLastMethodSuccess(HCkPkcs11 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC int CkPkcs11_getNumCerts(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getSharedLibPath(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkPkcs11_putSharedLibPath(HCkPkcs11 cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_sharedLibPath(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_getUtf8(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_putUtf8(HCkPkcs11 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_getVerboseLogging(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC void CkPkcs11_putVerboseLogging(HCkPkcs11 cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkPkcs11_getVersion(HCkPkcs11 cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkPkcs11_version(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_CloseSession(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_Discover(HCkPkcs11 cHandle, BOOL onlyTokensPresent, HCkJsonObject json);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_FindAllCerts(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_FindCert(HCkPkcs11 cHandle, const char *certPart, const char *partValue, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_GetCert(HCkPkcs11 cHandle, int index, HCkCert cert);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_Initialize(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_InitPin(HCkPkcs11 cHandle, const char *pin);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_InitToken(HCkPkcs11 cHandle, int slotId, const char *pin, const char *tokenLabel);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_Login(HCkPkcs11 cHandle, int userType, const char *pin);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_Logout(HCkPkcs11 cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_OpenSession(HCkPkcs11 cHandle, int slotId, BOOL readWrite);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_SaveLastError(HCkPkcs11 cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkPkcs11_SetPin(HCkPkcs11 cHandle, const char *oldPin, const char *newPin);
#endif
